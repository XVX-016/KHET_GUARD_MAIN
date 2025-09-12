import * as admin from 'firebase-admin';

let initialized = false;
export function initAdmin() {
  if (!initialized) {
    try {
      admin.initializeApp();
    } catch (_) {
      // ignore if already initialized
    }
    initialized = true;
  }
}

export function getFirestoreDb() {
  initAdmin();
  return admin.firestore();
}

export function getStorage() {
  initAdmin();
  return admin.storage();
}

