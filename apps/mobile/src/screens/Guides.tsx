import React from 'react';
import { View, Text, FlatList } from 'react-native';

const DATA = [
  { id: '1', title: 'Soil Health Basics' },
  { id: '2', title: 'Irrigation Best Practices' },
  { id: '3', title: 'Integrated Pest Management' }
];

export default function Guides() {
  return (
    <View style={{ flex: 1, padding: 16 }}>
      <FlatList
        data={DATA}
        keyExtractor={(i) => i.id}
        renderItem={({ item }) => (
          <View style={{ padding: 16, borderRadius: 12, backgroundColor: '#eee', marginBottom: 12 }}>
            <Text style={{ fontSize: 16 }}>{item.title}</Text>
          </View>
        )}
      />
    </View>
  );
}

