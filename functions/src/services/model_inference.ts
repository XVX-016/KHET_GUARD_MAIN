import { getFirestoreDb, getStorage } from '../utils/firestore';

type InferenceInput = {
  imageBase64?: string;
  storagePath?: string;
  metadata?: Record<string, unknown>;
};

export async function detectPlantDisease(input: InferenceInput) {
  // Placeholder: store image if provided, return mock result
  const storage = getStorage();
  let gsUri: string | undefined;
  if (input.imageBase64) {
    const buffer = Buffer.from(input.imageBase64, 'base64');
    const path = `uploads/plant/${Date.now()}.jpg`;
    await storage.bucket().file(path).save(buffer);
    gsUri = `gs://${storage.bucket().name}/${path}`;
  } else if (input.storagePath) {
    gsUri = input.storagePath;
  }
  // Return deterministic placeholder response
  return {
    diagnosis: 'Leaf_Blight_Placeholder',
    confidence: 0.72,
    suggested: ['Improve drainage', 'Apply copper fungicide as per label'],
    source: gsUri
  };
}

export async function detectCattleBreed(input: InferenceInput) {
  const storage = getStorage();
  let gsUri: string | undefined;
  if (input.imageBase64) {
    const buffer = Buffer.from(input.imageBase64, 'base64');
    const path = `uploads/cattle/${Date.now()}.jpg`;
    await storage.bucket().file(path).save(buffer);
    gsUri = `gs://${storage.bucket().name}/${path}`;
  } else if (input.storagePath) {
    gsUri = input.storagePath;
  }
  return {
    breed: 'Sahiwal_Placeholder',
    confidence: 0.65,
    tips: ['Provide clean water', 'Balanced fodder with minerals'],
    source: gsUri
  };
}

