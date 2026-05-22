import { createContext, useContext, useState } from 'react';
import uz from './uz';
import en from './en';
import ru from './ru';

const DICTS = { uz, en, ru };

export const LANGUAGES = [
  { code: 'uz', label: 'UZ', flag: '🇺🇿' },
  { code: 'en', label: 'EN', flag: '🇬🇧' },
  { code: 'ru', label: 'RU', flag: '🇷🇺' },
];

const LangContext = createContext(null);

export function LangProvider({ children }) {
  const [lang, setLang] = useState(
    () => localStorage.getItem('diabet_lang') || 'uz'
  );

  // Dot-notation lookup.
  // t('nav.patients')                  → translated string or key
  // t('patients.subtitle', {count: 5}) → with interpolation
  // t('biomarkers.hba1c', 'HbA1c')     → translated or fallback string
  const t = (key, varsOrFallback) => {
    const dict = DICTS[lang] || DICTS.uz;
    const value = key
      .split('.')
      .reduce((obj, k) => (obj && obj[k] !== undefined ? obj[k] : null), dict);

    if (value !== null) {
      if (varsOrFallback && typeof varsOrFallback === 'object') {
        return Object.entries(varsOrFallback).reduce(
          (s, [k, v]) => s.replace(`{${k}}`, v),
          value
        );
      }
      return value;
    }
    // key not found: return string fallback or the key itself
    return (typeof varsOrFallback === 'string') ? varsOrFallback : key;
  };

  const changeLang = (code) => {
    setLang(code);
    localStorage.setItem('diabet_lang', code);
  };

  return (
    <LangContext.Provider value={{ lang, t, changeLang }}>
      {children}
    </LangContext.Provider>
  );
}

export function useLang() {
  return useContext(LangContext);
}
