import React, { useMemo, useState } from 'react';
import { NavigationContainer, DefaultTheme, DarkTheme } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import HomeScreen from './screens/Home';
import GuidesScreen from './screens/Guides';
import ProfileScreen from './screens/Profile';
import CropRecommendationScreen from './screens/CropRecommendation';
import DiseaseScanScreen from './screens/DiseaseScan';
import CattleScanScreen from './screens/CattleScan';
import { initFirebase } from './services/firebase';
import './i18n';

initFirebase();

const Tab = createBottomTabNavigator();

export default function App() {
  const [dark, setDark] = useState(false);
  const theme = useMemo(() => (dark ? DarkTheme : DefaultTheme), [dark]);
  return (
    <NavigationContainer theme={theme}>
      <Tab.Navigator screenOptions={{ headerRight: () => null }}>
        <Tab.Screen name="Home" children={() => <HomeScreen onToggleTheme={() => setDark(v => !v)} />} />
        <Tab.Screen name="Guides" component={GuidesScreen} />
        <Tab.Screen name="CropRecommendation" component={CropRecommendationScreen} />
        <Tab.Screen name="DiseaseScan" component={DiseaseScanScreen} />
        <Tab.Screen name="CattleScan" component={CattleScanScreen} />
        <Tab.Screen name="Profile" component={ProfileScreen} />
      </Tab.Navigator>
    </NavigationContainer>
  );
}

