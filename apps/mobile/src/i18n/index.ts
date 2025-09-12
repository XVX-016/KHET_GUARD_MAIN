import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';

const resources = {
  en: { translation: { home: 'Home', guides: 'Guides', profile: 'Profile', recommend: 'Crop Recommendation', diseaseScan: 'Disease Scan', cattleScan: 'Cattle Scan', toggleTheme: 'Toggle Theme' } },
  hi: { translation: { home: 'होम', guides: 'मार्गदर्शन', profile: 'प्रोफ़ाइल', recommend: 'फसल सिफारिश', diseaseScan: 'रोग स्कैन', cattleScan: 'पशु स्कैन', toggleTheme: 'थीम बदलें' } }
};

i18n.use(initReactI18next).init({
  compatibilityJSON: 'v3',
  resources,
  lng: 'en',
  fallbackLng: 'en',
  interpolation: { escapeValue: false }
});

export default i18n;

