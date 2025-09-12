import React, { useState } from 'react';
import { View, Text, Button, Image } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

export default function CattleScan() {
  const [uri, setUri] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);

  async function pick() {
    const res = await ImagePicker.launchImageLibraryAsync({ base64: true, quality: 0.6 });
    if (!res.canceled) {
      const asset = res.assets?.[0];
      setUri(asset?.uri || null);
      const r = await fetch(process.env.EXPO_PUBLIC_FUNCTIONS_URL + '/detectCattle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ imageBase64: asset?.base64 })
      });
      setResult(await r.json());
    }
  }

  return (
    <View style={{ flex: 1, padding: 16 }}>
      <Button title="Pick Cattle Photo" onPress={pick} />
      {uri && <Image source={{ uri }} style={{ width: 200, height: 200, marginTop: 12 }} />}
      {result && (<Text style={{ marginTop: 12 }}>{JSON.stringify(result)}</Text>)}
    </View>
  );
}

