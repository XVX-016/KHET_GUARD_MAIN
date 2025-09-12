import React, { useState } from 'react';
import { View, Text, Button, ActivityIndicator, Modal } from 'react-native';
import * as Location from 'expo-location';
import LocationPicker from '../components/LocationPicker';

export default function CropRecommendation() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [manual, setManual] = useState(false);
  const [picked, setPicked] = useState<{lat:number,lon:number}|null>(null);

  async function fetchRec() {
    setLoading(true);
    try {
      let { status } = await Location.requestForegroundPermissionsAsync();
      let coords = { latitude: 28.6, longitude: 77.2 } as any;
      if (status === 'granted') {
        const loc = await Location.getCurrentPositionAsync({});
        coords = loc.coords;
      }
      if (picked) { coords = { latitude: picked.lat, longitude: picked.lon } as any; }
      const res = await fetch(process.env.EXPO_PUBLIC_FUNCTIONS_URL + '/recommendCrop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lat: coords.latitude, lon: coords.longitude })
      });
      setResult(await res.json());
    } finally {
      setLoading(false);
    }
  }

  return (
    <View style={{ flex: 1, padding: 16 }}>
      <Button title="Get Recommendation (GPS)" onPress={fetchRec} />
      <View style={{ height: 12 }} />
      <Button title="Pick on Map" onPress={() => setManual(true)} />
      {loading && <ActivityIndicator style={{ marginTop: 16 }} />}
      {result && (
        <View style={{ marginTop: 16 }}>
          <Text>Top Crops:</Text>
          {result.recommendations?.map((r: any) => (
            <Text key={r.crop}>{r.crop} — {r.score}</Text>
          ))}
        </View>
      )}
      <Modal visible={manual} animationType="slide">
        <LocationPicker onPick={(lat, lon) => { setPicked({lat,lon}); setManual(false); }} />
      </Modal>
    </View>
  );
}

