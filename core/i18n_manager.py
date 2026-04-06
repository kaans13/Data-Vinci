"""
i18n_manager.py — Data-Autopsy Uluslararasılaştırma Yöneticisi

Sorumluluklar:
  - Aktif dili yönetmek (tr / en)
  - UI bileşenlerini dil değişiminde otomatik güncellemek
  - Eksik anahtar durumunda fallback mantığı uygulamak
  - Flet page.update() döngüsüyle entegrasyon sağlamak
"""

from __future__ import annotations

import logging
from typing import Callable

from core.translations import TRANSLATIONS

logger = logging.getLogger(__name__)

# Desteklenen diller ve etiketleri
SUPPORTED_LANGUAGES: dict[str, str] = {
    "tr": "Türkçe",
    "en": "English",
}

DEFAULT_LANGUAGE = "tr"
FALLBACK_LANGUAGE = "en"


class I18nManager:
    """
    Merkezi dil yöneticisi.

    Kullanım:
        i18n = I18nManager()
        i18n.set_language("en")
        label = i18n.t("menu_normalize")  # → "Normalize"

    Dil değişim olayları:
        i18n.on_language_change(callback) ile dinleyici eklenebilir.
        Flet page.update() çağrısı callback içinde yapılmalıdır.
    """

    def __init__(self, initial_language: str = DEFAULT_LANGUAGE) -> None:
        self._language: str = initial_language
        self._listeners: list[Callable[[str], None]] = []

        if initial_language not in SUPPORTED_LANGUAGES:
            logger.warning(
                "Desteklenmeyen dil '%s', varsayılan '%s' kullanılıyor.",
                initial_language,
                DEFAULT_LANGUAGE,
            )
            self._language = DEFAULT_LANGUAGE

    # ------------------------------------------------------------------
    # Çeviri
    # ------------------------------------------------------------------

    def t(self, key: str, **kwargs) -> str:
        """
        Verilen anahtarın aktif dildeki çevirisini döner.

        Öncelik sırası:
          1. Aktif dil sözlüğü
          2. Fallback dil (en)
          3. Anahtarın kendisi (kayıp durumu için güvenli fallback)

        kwargs ile dinamik doldurma:
          i18n.t("rows_loaded", n=1500)  # "1500 rows loaded"
        """
        primary = TRANSLATIONS.get(self._language, {})
        text = primary.get(key)

        if text is None:
            fallback = TRANSLATIONS.get(FALLBACK_LANGUAGE, {})
            text = fallback.get(key)
            if text is not None:
                logger.debug("Fallback kullanıldı: key='%s' lang='%s'", key, self._language)
            else:
                logger.warning("Çeviri bulunamadı: key='%s'", key)
                return key  # En kötü durumda key'i göster

        # Dinamik parametre doldurma
        if kwargs:
            try:
                text = text.format(**kwargs)
            except KeyError as e:
                logger.error("Çeviri formatı hatası key='%s': %s", key, e)

        return text

    # ------------------------------------------------------------------
    # Dil Yönetimi
    # ------------------------------------------------------------------

    @property
    def language(self) -> str:
        """Aktif dil kodu (tr, en, ...)."""
        return self._language

    @property
    def language_label(self) -> str:
        """Aktif dilin okunabilir etiketi."""
        return SUPPORTED_LANGUAGES.get(self._language, self._language.upper())

    def set_language(self, lang_code: str) -> bool:
        """
        Dili değiştirir ve kayıtlı tüm dinleyicileri tetikler.

        Returns:
            True  → dil başarıyla değiştirildi
            False → geçersiz dil kodu
        """
        if lang_code not in SUPPORTED_LANGUAGES:
            logger.error("Geçersiz dil kodu: '%s'", lang_code)
            return False

        if lang_code == self._language:
            return True  # Zaten aktif

        old_lang = self._language
        self._language = lang_code
        logger.info("Dil değiştirildi: %s → %s", old_lang, lang_code)

        # Tüm dinleyicileri bildir (Flet page.update() buradan tetiklenir)
        self._notify_listeners(lang_code)
        return True

    def toggle_language(self) -> str:
        """
        İki dil arasında geçiş yapar (tr ↔ en).
        Yeni aktif dil kodunu döner.
        """
        langs = list(SUPPORTED_LANGUAGES.keys())
        current_idx = langs.index(self._language) if self._language in langs else 0
        next_lang = langs[(current_idx + 1) % len(langs)]
        self.set_language(next_lang)
        return next_lang

    # ------------------------------------------------------------------
    # Dinleyici Yönetimi
    # ------------------------------------------------------------------

    def on_language_change(self, callback: Callable[[str], None]) -> None:
        """
        Dil değişim olayına dinleyici ekler.

        Callback imzası: callback(new_lang_code: str) -> None

        Örnek (Flet ile):
            def refresh_ui(lang):
                header.value = i18n.t("app_title")
                page.update()

            i18n.on_language_change(refresh_ui)
        """
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_language_listener(self, callback: Callable[[str], None]) -> None:
        """Dinleyiciyi kaldırır (bellek yönetimi için)."""
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _notify_listeners(self, new_lang: str) -> None:
        """Tüm kayıtlı dinleyicileri yeni dil koduyla çağırır."""
        for cb in self._listeners:
            try:
                cb(new_lang)
            except Exception as exc:
                logger.exception("Dil dinleyicisi hatası: %s", exc)

    # ------------------------------------------------------------------
    # Yardımcı
    # ------------------------------------------------------------------

    def get_all_keys(self) -> list[str]:
        """Tüm çeviri anahtarlarını listeler (debug/test için)."""
        return list(TRANSLATIONS.get(DEFAULT_LANGUAGE, {}).keys())

    def get_missing_keys(self, lang: str) -> list[str]:
        """
        Belirtilen dilde eksik anahtarları bulur.
        Çeviri bütünlüğü doğrulaması için kullanılır.
        """
        base_keys = set(TRANSLATIONS.get(DEFAULT_LANGUAGE, {}).keys())
        lang_keys = set(TRANSLATIONS.get(lang, {}).keys())
        missing = sorted(base_keys - lang_keys)
        if missing:
            logger.warning("'%s' dilinde eksik anahtarlar: %s", lang, missing)
        return missing

    def validate_translations(self) -> dict[str, list[str]]:
        """
        Tüm dillerin çeviri bütünlüğünü doğrular.
        Returns: {lang_code: [eksik_anahtarlar]}
        """
        result = {}
        for lang in SUPPORTED_LANGUAGES:
            missing = self.get_missing_keys(lang)
            if missing:
                result[lang] = missing
        return result

    def __repr__(self) -> str:
        return f"I18nManager(language='{self._language}', listeners={len(self._listeners)})"
