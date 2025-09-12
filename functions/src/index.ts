import * as functions from 'firebase-functions';
import express from 'express';
import multer from 'multer';
import { getSoilFeatures } from './services/soilgrids';
import { getWeatherFeatures } from './services/openweather';
import { getNdviSummary } from './services/gee';
import { getFirestoreDb } from './utils/firestore';
import { recommendCrops } from './services/recommender';
import { detectPlantDisease, detectCattleBreed } from './services/model_inference';

const app = express();
app.use(express.json({ limit: '10mb' }));
const upload = multer();

type LatLon = { lat: number; lon: number };

async function resolveLocation(input: any): Promise<LatLon> {
  if (typeof input?.lat === 'number' && typeof input?.lon === 'number') {
    return { lat: input.lat, lon: input.lon };
  }
  if (input?.locationId) {
    const db = getFirestoreDb();
    const doc = await db.collection('locations').doc(input.locationId).get();
    const data = doc.data();
    if (!data) throw new Error('Location not found');
    return { lat: data.lat, lon: data.lon } as LatLon;
  }
  throw new Error('lat/lon or locationId required');
}

app.post('/recommendCrop', async (req, res) => {
  try {
    const { lat, lon } = await resolveLocation(req.body);
    const db = getFirestoreDb();
    const cacheRef = db.collection('location_data_cache').doc(`${lat}_${lon}`);
    const cacheSnap = await cacheRef.get();
    let ndviSummary: any = cacheSnap.exists ? cacheSnap.data()?.ndviSummary : null;
    if (!ndviSummary) {
      ndviSummary = await getNdviSummary(lat, lon);
      await cacheRef.set({ ndviSummary, updatedAt: Date.now() }, { merge: true });
    }
    const [soil, weather] = await Promise.all([
      getSoilFeatures(lat, lon),
      getWeatherFeatures(lat, lon)
    ]);
    const features = { lat, lon, soil, weather, ndvi: ndviSummary };
    const recommendations = recommendCrops(features);
    res.json({ features, recommendations });
  } catch (err: any) {
    res.status(400).json({ error: err.message || 'Unknown error' });
  }
});

app.post('/detectPlantDisease', upload.none(), async (req, res) => {
  try {
    const { imageBase64, storagePath, metadata } = req.body || {};
    const result = await detectPlantDisease({ imageBase64, storagePath, metadata });
    res.json(result);
  } catch (err: any) {
    res.status(400).json({ error: err.message || 'Unknown error' });
  }
});

app.post('/detectCattle', upload.none(), async (req, res) => {
  try {
    const { imageBase64, storagePath, metadata } = req.body || {};
    const result = await detectCattleBreed({ imageBase64, storagePath, metadata });
    res.json(result);
  } catch (err: any) {
    res.status(400).json({ error: err.message || 'Unknown error' });
  }
});

app.get('/location/:id/data', async (req, res) => {
  try {
    const db = getFirestoreDb();
    const loc = await db.collection('locations').doc(req.params.id).get();
    const data = loc.data();
    if (!data) return res.status(404).json({ error: 'Not found' });
    const cache = await db
      .collection('location_data_cache')
      .doc(`${data.lat}_${data.lon}`)
      .get();
    res.json({ location: data, cache: cache.data() || {} });
  } catch (err: any) {
    res.status(400).json({ error: err.message || 'Unknown error' });
  }
});

export const api = functions.https.onRequest(app);
