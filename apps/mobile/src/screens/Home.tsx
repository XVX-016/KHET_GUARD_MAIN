import React from 'react';
import { View, Text, Button } from 'react-native';
import LottieView from 'lottie-react-native';
import { useTranslation } from 'react-i18next';

export default function Home({ onToggleTheme }: { onToggleTheme: () => void }) {
  const { t, i18n } = useTranslation();
  return (
    <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center', padding: 16 }}>
      <LottieView source={require('../assets/3d/plant.json')} autoPlay loop style={{ width: 200, height: 200 }} />
      <Text style={{ fontSize: 24, marginBottom: 12 }}>Khet Guard</Text>
      <View style={{ flexDirection: 'row', gap: 8 }}>
        <Button title={t('toggleTheme')} onPress={onToggleTheme} />
        <Button title="EN" onPress={() => i18n.changeLanguage('en')} />
        <Button title="हिंदी" onPress={() => i18n.changeLanguage('hi')} />
      </View>
    </View>
  );
}

