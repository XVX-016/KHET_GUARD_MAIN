import React, { useState, useEffect } from 'react';
import { View, Text, Button, Image, ActivityIndicator, ScrollView, StyleSheet, Alert } from 'react-native';
import { useTranslation } from 'react-i18next';
import ImageProcessor from '../services/imageProcessor';
import ErrorHandler from '../services/errorHandler';
import NetInfo from '@react-native-community/netinfo';

interface PredictionResult {
  predictions: Array<{
    class: string;
    confidence: number;
    class_id: number;
  }>;
  model_info: {
    model_type: string;
    version: string;
    num_classes: number;
  };
  processing_time: number;
  uncertainty?: {
    entropy: number;
    max_uncertainty: number;
    confidence: number;
  };
}

export default function DiseaseScan() {
  const { t } = useTranslation();
  const [image, setImage] = useState<{ uri: string; base64: string } | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [isOnline, setIsOnline] = useState(true);

  useEffect(() => {
    // Check network connectivity
    const unsubscribe = NetInfo.addEventListener(state => {
      setIsOnline(state.isConnected ?? false);
    });

    // Load queued uploads
    ErrorHandler.loadUploadQueue();

    return unsubscribe;
  }, []);

  const captureImage = async () => {
    try {
      setLoading(true);
      const processedImage = await ImageProcessor.captureImage({
        quality: 0.7,
        maxWidth: 1024,
        maxHeight: 1024,
        minWidth: 256,
        minHeight: 256,
      });

      if (processedImage) {
        setImage({
          uri: processedImage.uri,
          base64: processedImage.base64,
        });
        setResult(null);
      }
    } catch (error) {
      ErrorHandler.handleError(error as Error, {
        operation: 'capture_image',
        timestamp: Date.now(),
      });
    } finally {
      setLoading(false);
    }
  };

  const selectImage = async () => {
    try {
      setLoading(true);
      const processedImage = await ImageProcessor.selectImage({
        quality: 0.7,
        maxWidth: 1024,
        maxHeight: 1024,
        minWidth: 256,
        minHeight: 256,
      });

      if (processedImage) {
        setImage({
          uri: processedImage.uri,
          base64: processedImage.base64,
        });
        setResult(null);
      }
    } catch (error) {
      ErrorHandler.handleError(error as Error, {
        operation: 'select_image',
        timestamp: Date.now(),
      });
    } finally {
      setLoading(false);
    }
  };

  const analyzeImage = async () => {
    if (!image) return;

    try {
      setLoading(true);

      if (!isOnline) {
        // Queue for upload when online
        await ErrorHandler.queueUpload({
          imageUri: image.uri,
          imageBase64: image.base64,
          metadata: {
            lat: 0, // Would get from location service
            lon: 0,
            timestamp: Date.now(),
            operation: 'plant_disease',
          },
        });
        return;
      }

      const response = await fetch(`${process.env.EXPO_PUBLIC_FUNCTIONS_URL}/predict/plant_disease`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${await getAuthToken()}`,
        },
        body: JSON.stringify({
          imageBase64: image.base64,
          return_uncertainty: true,
        }),
      });

      if (!response.ok) {
        throw new Error(`Analysis failed with status ${response.status}`);
      }

      const analysisResult = await response.json();
      setResult(analysisResult);
    } catch (error) {
      ErrorHandler.handleError(error as Error, {
        operation: 'analyze_image',
        timestamp: Date.now(),
        metadata: { hasImage: !!image },
      });
    } finally {
      setLoading(false);
    }
  };

  const getAuthToken = async (): Promise<string> => {
    // Implement your auth token retrieval logic
    return 'mock_token';
  };

  const getConfidenceColor = (confidence: number): string => {
    if (confidence >= 0.8) return '#4CAF50'; // Green
    if (confidence >= 0.6) return '#FF9800'; // Orange
    return '#F44336'; // Red
  };

  const getUncertaintyLevel = (uncertainty?: { entropy: number; confidence: number }): string => {
    if (!uncertainty) return 'Unknown';
    if (uncertainty.confidence >= 0.8) return 'Low';
    if (uncertainty.confidence >= 0.6) return 'Medium';
    return 'High';
  };

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>Plant Disease Detection</Text>
      
      {!isOnline && (
        <View style={styles.offlineBanner}>
          <Text style={styles.offlineText}>You're offline. Images will be queued for upload.</Text>
        </View>
      )}

      <View style={styles.buttonContainer}>
        <Button
          title="Take Photo"
          onPress={captureImage}
          disabled={loading}
        />
        <View style={styles.buttonSpacer} />
        <Button
          title="Select from Gallery"
          onPress={selectImage}
          disabled={loading}
        />
      </View>

      {loading && (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#2196F3" />
          <Text style={styles.loadingText}>Processing...</Text>
        </View>
      )}

      {image && (
        <View style={styles.imageContainer}>
          <Image source={{ uri: image.uri }} style={styles.image} />
          <Button
            title="Analyze Disease"
            onPress={analyzeImage}
            disabled={loading}
            color="#4CAF50"
          />
        </View>
      )}

      {result && (
        <View style={styles.resultContainer}>
          <Text style={styles.resultTitle}>Analysis Results</Text>
          
          <View style={styles.modelInfo}>
            <Text style={styles.modelText}>
              Model: {result.model_info.model_type} v{result.model_info.version}
            </Text>
            <Text style={styles.modelText}>
              Processing time: {result.processing_time.toFixed(2)}s
            </Text>
          </View>

          <Text style={styles.predictionsTitle}>Top Predictions:</Text>
          {result.predictions.map((pred, index) => (
            <View key={index} style={styles.predictionItem}>
              <View style={styles.predictionHeader}>
                <Text style={styles.predictionClass}>{pred.class}</Text>
                <Text style={[
                  styles.predictionConfidence,
                  { color: getConfidenceColor(pred.confidence) }
                ]}>
                  {(pred.confidence * 100).toFixed(1)}%
                </Text>
              </View>
              <View style={styles.confidenceBar}>
                <View 
                  style={[
                    styles.confidenceFill,
                    { 
                      width: `${pred.confidence * 100}%`,
                      backgroundColor: getConfidenceColor(pred.confidence)
                    }
                  ]} 
                />
              </View>
            </View>
          ))}

          {result.uncertainty && (
            <View style={styles.uncertaintyContainer}>
              <Text style={styles.uncertaintyTitle}>Uncertainty Analysis:</Text>
              <Text style={styles.uncertaintyText}>
                Level: {getUncertaintyLevel(result.uncertainty)}
              </Text>
              <Text style={styles.uncertaintyText}>
                Confidence: {(result.uncertainty.confidence * 100).toFixed(1)}%
              </Text>
              <Text style={styles.uncertaintyText}>
                Entropy: {result.uncertainty.entropy.toFixed(3)}
              </Text>
            </View>
          )}

          <View style={styles.recommendationsContainer}>
            <Text style={styles.recommendationsTitle}>Recommendations:</Text>
            <Text style={styles.recommendationsText}>
              • Take clear, well-lit photos for better accuracy
            </Text>
            <Text style={styles.recommendationsText}>
              • Focus on the affected area of the plant
            </Text>
            <Text style={styles.recommendationsText}>
              • Avoid blurry or shadowy images
            </Text>
          </View>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 20,
    color: '#333',
  },
  offlineBanner: {
    backgroundColor: '#FF9800',
    padding: 12,
    borderRadius: 8,
    marginBottom: 16,
  },
  offlineText: {
    color: 'white',
    textAlign: 'center',
    fontWeight: '500',
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 20,
  },
  buttonSpacer: {
    width: 16,
  },
  loadingContainer: {
    alignItems: 'center',
    padding: 20,
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: '#666',
  },
  imageContainer: {
    alignItems: 'center',
    marginBottom: 20,
  },
  image: {
    width: 200,
    height: 200,
    borderRadius: 10,
    marginBottom: 16,
  },
  resultContainer: {
    backgroundColor: 'white',
    padding: 16,
    borderRadius: 10,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  resultTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 16,
    color: '#333',
  },
  modelInfo: {
    backgroundColor: '#f0f0f0',
    padding: 12,
    borderRadius: 8,
    marginBottom: 16,
  },
  modelText: {
    fontSize: 14,
    color: '#666',
    marginBottom: 4,
  },
  predictionsTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 12,
    color: '#333',
  },
  predictionItem: {
    marginBottom: 12,
  },
  predictionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  predictionClass: {
    fontSize: 16,
    fontWeight: '500',
    color: '#333',
  },
  predictionConfidence: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  confidenceBar: {
    height: 8,
    backgroundColor: '#e0e0e0',
    borderRadius: 4,
    overflow: 'hidden',
  },
  confidenceFill: {
    height: '100%',
    borderRadius: 4,
  },
  uncertaintyContainer: {
    backgroundColor: '#f8f9fa',
    padding: 12,
    borderRadius: 8,
    marginTop: 16,
  },
  uncertaintyTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 8,
    color: '#333',
  },
  uncertaintyText: {
    fontSize: 14,
    color: '#666',
    marginBottom: 4,
  },
  recommendationsContainer: {
    marginTop: 16,
    padding: 12,
    backgroundColor: '#e8f5e8',
    borderRadius: 8,
  },
  recommendationsTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 8,
    color: '#2e7d32',
  },
  recommendationsText: {
    fontSize: 14,
    color: '#388e3c',
    marginBottom: 4,
  },
});

