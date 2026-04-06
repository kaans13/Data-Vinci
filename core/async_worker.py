"""
async_worker.py — Data-Autopsy v5  Doğru Async/Thread Mimarisi

Problem
=======
Şu an: `threading.Thread(target=heavy_fn, daemon=True)` → thread'den
`page.update()` çağrısı yapıyoruz.

Neden bu tehlikeli?
  - Flet, asyncio event loop üzerinde çalışır
  - `page.update()` asyncio coroutine'i tetikler
  - Thread'den coroutine çağırmak → `RuntimeError: This event loop is already running`
    veya segmentation fault (özellikle Flet web modunda)
  - CPython GIL CPU-bound işlemleri zaten serialize eder; thread kazancı yok

Doğru Çözüm: İki Senaryo
=========================

Senaryo A — I/O Bound (dosya okuma, ağ, DuckDB sorgu):
  → `asyncio.get_event_loop().run_in_executor(ThreadPoolExecutor)`
  → Async fonksiyon, event loop'u bloklamaz
  → `page.run_task(coro)` ile UI güvenli güncelleme

Senaryo B — CPU Bound (büyük veri normalizasyonu, imputation, istatistik):
  → `ProcessPoolExecutor` — ayrı Python process, GIL yok
  → İşlem bitti → `asyncio.wrap_future()` ile await edilebilir hale gelir
  → Sonuç ana process'e döner, UI güncellenir

Ama bekle: Flet genellikle sync callback kullanır (on_click, on_change).
Sync callback içinden async başlatmak için `page.run_task()` var.

Hangisini ne zaman kullan?
  - Flet sync callback → WorkerBridge.run_sync() → page.run_task() → async worker
  - Flet async callback → doğrudan await worker
  - CPU-heavy (>1s) → ProcessPoolExecutor
  - I/O-heavy → ThreadPoolExecutor (asyncio executor)

Neden threading.Thread tamamen kötü değil?
  Flet'in desktop (non-web) modunda threading.Thread + page.update()
  genellikle çalışır ÇÜNKİ Flet desktop'ta sync bağlamda update() kabul eder.
  Ama bu bir uygulama detayı, garanti değil. Web modunda kırılır.
  Doğru yol: run_task() köprüsünü kullanmak.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, TypeVar

import flet as ft

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Global executor'lar — uygulama başlangıcında bir kez oluşturulur
_THREAD_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="da_io")
_PROCESS_POOL: ProcessPoolExecutor | None = None   # lazy init


def get_process_pool() -> ProcessPoolExecutor:
    global _PROCESS_POOL
    if _PROCESS_POOL is None:
        # CPU sayısına göre worker, max 4
        import os
        workers = min(4, max(1, (os.cpu_count() or 2) - 1))
        _PROCESS_POOL = ProcessPoolExecutor(max_workers=workers)
        logger.info("ProcessPoolExecutor başlatıldı (%d worker)", workers)
    return _PROCESS_POOL


def shutdown_pools() -> None:
    """Uygulama kapanırken çağır."""
    _THREAD_POOL.shutdown(wait=False)
    if _PROCESS_POOL:
        _PROCESS_POOL.shutdown(wait=False)


# ── Temel Worker Sınıfı ──────────────────────────────────────────────────────

class AsyncWorker:
    """
    I/O-bound ve CPU-bound işlemler için unified interface.

    Kullanım (Flet sync on_click içinden):
        worker = AsyncWorker(page)
        worker.run_io(
            fn       = lambda: db.load_file(path),
            on_done  = lambda result: update_ui(result),
            on_error = lambda exc: show_error(str(exc)),
            title    = "Dosya yükleniyor...",
        )

    Kullanım (Flet async callback içinden):
        result = await worker.await_io(lambda: db.load_file(path))
    """

    def __init__(self, page: ft.Page):
        self.page = page

    # ── I/O-bound ────────────────────────────────────────────────────────

    def run_io(
        self,
        fn: Callable[[], T],
        on_done: Callable[[T], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
        title: str = "İşleniyor...",
    ) -> None:
        """
        Sync callback'ten I/O işlemi başlatır.
        `page.run_task()` ile event loop'a teslim eder → UI donmaz.
        """
        async def _coro():
            loop   = asyncio.get_event_loop()
            t0     = time.perf_counter()
            try:
                result = await loop.run_in_executor(_THREAD_POOL, fn)
                dur    = (time.perf_counter() - t0) * 1000
                logger.info("[IO] %s tamamlandı %.0fms", title, dur)
                if on_done:
                    on_done(result)
            except Exception as exc:
                logger.exception("[IO] %s hatası", title)
                if on_error:
                    on_error(exc)

        self.page.run_task(_coro)

    async def await_io(self, fn: Callable[[], T]) -> T:
        """Async context'ten I/O işlemi bekler."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_THREAD_POOL, fn)

    # ── CPU-bound ────────────────────────────────────────────────────────

    def run_cpu(
        self,
        fn: Callable[[], T],
        on_done: Callable[[T], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
        title: str = "Hesaplanıyor...",
    ) -> None:
        """
        CPU-bound iş için ProcessPoolExecutor kullanır.

        Sınırlama: fn ve dönüş değeri pickle edilebilir olmalı
        (lambda, closure, DataFrame → OK; flet nesneleri → HAYIR).
        """
        async def _coro():
            loop = asyncio.get_event_loop()
            pool = get_process_pool()
            t0   = time.perf_counter()
            try:
                result = await loop.run_in_executor(pool, fn)
                dur    = (time.perf_counter() - t0) * 1000
                logger.info("[CPU] %s tamamlandı %.0fms", title, dur)
                if on_done:
                    on_done(result)
            except Exception as exc:
                logger.exception("[CPU] %s hatası", title)
                if on_error:
                    on_error(exc)

        self.page.run_task(_coro)

    async def await_cpu(self, fn: Callable[[], T]) -> T:
        """Async context'ten CPU işlemi bekler."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(get_process_pool(), fn)


# ── Uyumluluk Köprüsü (mevcut kod migration için) ───────────────────────────

class WorkerBridge:
    """
    Mevcut `LoadingOverlay.run()` API'sini koruyarak
    altını AsyncWorker ile değiştirir.

    Migration: LoadingOverlay.run() → WorkerBridge.run()
    Kod değişikliği yok; sadece nesneyi değiştir.
    """

    def __init__(self, page: ft.Page, overlay=None):
        self.page    = page
        self.overlay = overlay
        self._worker = AsyncWorker(page)

    def run(
        self,
        title: str,
        fn: Callable[[], T],
        on_done: Callable[[T], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
        cpu_bound: bool = False,
    ) -> None:
        """
        Mevcut API ile uyumlu:
            loading.run("Yükleniyor", heavy_fn, on_done, on_error)
        """
        if self.overlay:
            self.overlay.show(title)

        def _wrap_done(result):
            if self.overlay:
                self.overlay.hide()
            if on_done:
                on_done(result)

        def _wrap_err(exc):
            if self.overlay:
                self.overlay.hide()
            if on_error:
                on_error(exc)

        if cpu_bound:
            self._worker.run_cpu(fn, _wrap_done, _wrap_err, title)
        else:
            self._worker.run_io(fn, _wrap_done, _wrap_err, title)

    def set_progress(self, pct: float, msg: str = "") -> None:
        if self.overlay:
            self.overlay.set_progress(pct, msg)


# ── Büyük Veri Stratejisi ────────────────────────────────────────────────────

class ChunkedProcessor:
    """
    Büyük DataFrame'leri parçalara bölerek işler.
    Her chunk'tan sonra UI'a ilerleme bildirir.

    Kullanım:
        processor = ChunkedProcessor(page, chunk_size=10_000)
        result = await processor.process(
            df=big_df,
            fn=normalize_chunk,      # (chunk_df) → chunk_df
            on_progress=update_bar,  # (pct, msg) → None
        )
    """

    def __init__(self, page: ft.Page, chunk_size: int = 10_000):
        self.page       = page
        self.chunk_size = chunk_size
        self._worker    = AsyncWorker(page)

    async def process(
        self,
        df,
        fn: Callable,
        on_progress: Callable[[float, str], None] | None = None,
    ):
        """
        Chunk'lara böl → her chunk'ı thread pool'da işle → birleştir.
        """
        import pandas as pd

        n      = len(df)
        chunks = [df.iloc[i:i + self.chunk_size].copy()
                  for i in range(0, n, self.chunk_size)]
        total  = len(chunks)
        results = []

        for idx, chunk in enumerate(chunks):
            # Her chunk için thread pool
            result = await self._worker.await_io(lambda c=chunk: fn(c))
            results.append(result)

            pct = (idx + 1) / total * 100
            msg = f"{idx + 1}/{total} parça işlendi"
            if on_progress:
                on_progress(pct, msg)
            # Event loop'a nefes aldır → UI güncellenir
            await asyncio.sleep(0)

        return pd.concat(results, ignore_index=True)
