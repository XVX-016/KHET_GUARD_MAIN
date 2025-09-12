import React, { useState } from 'react';
import { View, Button } from 'react-native';
import MapView, { Marker, MapPressEvent } from 'react-native-maps';

export default function LocationPicker({ onPick }: { onPick: (lat: number, lon: number) => void }) {
  const [coord, setCoord] = useState({ latitude: 28.6, longitude: 77.2 });
  function onPress(e: MapPressEvent) {
    const { latitude, longitude } = e.nativeEvent.coordinate;
    setCoord({ latitude, longitude });
  }
  return (
    <View style={{ flex: 1 }}>
      <MapView style={{ flex: 1 }} initialRegion={{ latitude: coord.latitude, longitude: coord.longitude, latitudeDelta: 0.2, longitudeDelta: 0.2 }} onPress={onPress}>
        <Marker coordinate={coord} />
      </MapView>
      <View style={{ padding: 12 }}>
        <Button title="Use This Location" onPress={() => onPick(coord.latitude, coord.longitude)} />
      </View>
    </View>
  );
}

