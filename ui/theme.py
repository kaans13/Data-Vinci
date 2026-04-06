"""
theme.py — Data-Autopsy tema sistemi

Renk sabitlerini module-level değil, çağrılabilir fonksiyon olarak sunar.
Böylece tema değişiminde tüm widget'lar doğru renkleri alır.
"""
from __future__ import annotations

# Sabit aksanlar — temadan bağımsız
P    = "#6C63FF"   # primary purple
S    = "#4CAF50"   # success green
W    = "#FF9800"   # warning amber
E    = "#F44336"   # error red
IC   = "#29B6F6"   # info cyan
GOLD = "#FFD700"

# Aktif tema referansı (mutable dict — views/main bunu okur)
_T: dict[str, str] = {}

DARK: dict[str, str] = {
    "BG":   "#09090F",
    "NAV":  "#0D0D1E",
    "CARD": "#13132A",
    "CARD2":"#1A1A30",
    "TEXT": "#E8E8F0",
    "SUB":  "#6666AA",
    "BDR":  "#222244",
    "ICON": "#9999CC",
}

LIGHT: dict[str, str] = {
    "BG":   "#F4F4FB",
    "NAV":  "#FFFFFF",
    "CARD": "#FFFFFF",
    "CARD2":"#F0F0FA",
    "TEXT": "#1A1A2E",
    "SUB":  "#666688",
    "BDR":  "#DADAEE",
    "ICON": "#444466",
}

def set_theme(dark: bool) -> None:
    src = DARK if dark else LIGHT
    _T.clear()
    _T.update(src)

def get(key: str) -> str:
    return _T.get(key, DARK.get(key, "#000000"))

# Kısayollar — her zaman güncel değeri döner
def BG()   -> str: return _T.get("BG",   DARK["BG"])
def NAV()  -> str: return _T.get("NAV",  DARK["NAV"])
def CARD() -> str: return _T.get("CARD", DARK["CARD"])
def CARD2()-> str: return _T.get("CARD2",DARK["CARD2"])
def TEXT() -> str: return _T.get("TEXT", DARK["TEXT"])
def SUB()  -> str: return _T.get("SUB",  DARK["SUB"])
def BDR()  -> str: return _T.get("BDR",  DARK["BDR"])

# Başlangıçta dark yükle
set_theme(True)
