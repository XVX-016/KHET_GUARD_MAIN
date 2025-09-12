/**
 * Centralized error handling service for Khet Guard mobile app.
 * Handles network errors, upload failures, and provides user-friendly messages.
 */

import { Alert, NetInfo } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as Sentry from '@sentry/react-native';

export interface ErrorContext {
  operation: string;
  userId?: string;
  timestamp: number;
  metadata?: Record<string, any>;
}

export interface QueuedUpload {
  id: string;
  imageUri: string;
  imageBase64: string;
  metadata: {
    lat: number;
    lon: number;
    crop?: string;
    timestamp: number;
    operation: 'plant_disease' | 'cattle_breed';
  };
  retryCount: number;
  maxRetries: number;
}

class ErrorHandler {
  private static instance: ErrorHandler;
  private uploadQueue: QueuedUpload[] = [];
  private isProcessingQueue = false;
  private retryDelays = [2000, 4000, 8000]; // Exponential backoff delays

  static getInstance(): ErrorHandler {
    if (!ErrorHandler.instance) {
      ErrorHandler.instance = new ErrorHandler();
    }
    return ErrorHandler.instance;
  }

  /**
   * Handle errors with user-friendly messages and logging
   */
  handleError(error: Error, context: ErrorContext): void {
    // Log to Sentry
    Sentry.captureException(error, {
      tags: {
        operation: context.operation,
        userId: context.userId,
      },
      extra: {
        timestamp: context.timestamp,
        metadata: context.metadata,
      },
    });

    // Determine user-friendly message
    const userMessage = this.getUserFriendlyMessage(error, context.operation);
    
    // Show alert to user
    Alert.alert(
      'Error',
      userMessage,
      [
        {
          text: 'OK',
          style: 'default',
        },
        ...(this.shouldOfferRetry(error) ? [{
          text: 'Retry',
          onPress: () => this.handleRetry(context),
        }] : []),
      ]
    );
  }

  /**
   * Handle network connectivity changes
   */
  async handleConnectivityChange(): Promise<void> {
    const state = await NetInfo.fetch();
    
    if (state.isConnected && this.uploadQueue.length > 0) {
      console.log('Network connected, processing upload queue...');
      await this.processUploadQueue();
    }
  }

  /**
   * Queue upload for retry when network is available
   */
  async queueUpload(upload: Omit<QueuedUpload, 'id' | 'retryCount' | 'maxRetries'>): Promise<void> {
    const queuedUpload: QueuedUpload = {
      ...upload,
      id: `upload_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      retryCount: 0,
      maxRetries: 3,
    };

    this.uploadQueue.push(queuedUpload);
    await this.saveUploadQueue();

    // Try to process immediately if online
    const state = await NetInfo.fetch();
    if (state.isConnected) {
      await this.processUploadQueue();
    } else {
      Alert.alert(
        'Upload Queued',
        'Your image has been saved and will be uploaded when you have internet connection.',
        [{ text: 'OK' }]
      );
    }
  }

  /**
   * Process queued uploads
   */
  private async processUploadQueue(): Promise<void> {
    if (this.isProcessingQueue || this.uploadQueue.length === 0) {
      return;
    }

    this.isProcessingQueue = true;

    try {
      const uploads = [...this.uploadQueue];
      this.uploadQueue = [];

      for (const upload of uploads) {
        try {
          await this.uploadImage(upload);
          console.log(`Successfully uploaded ${upload.id}`);
        } catch (error) {
          console.error(`Failed to upload ${upload.id}:`, error);
          
          // Retry if under max retries
          if (upload.retryCount < upload.maxRetries) {
            upload.retryCount++;
            this.uploadQueue.push(upload);
          } else {
            console.error(`Max retries exceeded for ${upload.id}`);
            // Could notify user of permanent failure
          }
        }

        // Add delay between uploads
        await new Promise(resolve => setTimeout(resolve, 1000));
      }

      await this.saveUploadQueue();
    } finally {
      this.isProcessingQueue = false;
    }
  }

  /**
   * Upload image to server
   */
  private async uploadImage(upload: QueuedUpload): Promise<void> {
    const { imageBase64, metadata, operation } = upload;
    
    const endpoint = operation === 'plant_disease' 
      ? '/predict/plant_disease' 
      : '/predict/cattle_breed';

    const response = await fetch(`${process.env.EXPO_PUBLIC_FUNCTIONS_URL}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${await this.getAuthToken()}`,
      },
      body: JSON.stringify({
        imageBase64,
        metadata,
        return_uncertainty: true,
      }),
    });

    if (!response.ok) {
      throw new Error(`Upload failed with status ${response.status}`);
    }

    const result = await response.json();
    console.log('Upload result:', result);
  }

  /**
   * Get user-friendly error message
   */
  private getUserFriendlyMessage(error: Error, operation: string): string {
    const errorMessage = error.message.toLowerCase();

    if (errorMessage.includes('network') || errorMessage.includes('fetch')) {
      return 'Please check your internet connection and try again.';
    }

    if (errorMessage.includes('permission') || errorMessage.includes('camera')) {
      return 'Camera permission is required. Please enable it in your device settings.';
    }

    if (errorMessage.includes('image') || errorMessage.includes('format')) {
      return 'Invalid image format. Please take a clear photo and try again.';
    }

    if (errorMessage.includes('size') || errorMessage.includes('large')) {
      return 'Image is too large. Please take a smaller photo and try again.';
    }

    if (errorMessage.includes('timeout')) {
      return 'Request timed out. Please check your connection and try again.';
    }

    if (errorMessage.includes('unauthorized') || errorMessage.includes('401')) {
      return 'Please log in again to continue.';
    }

    if (errorMessage.includes('server') || errorMessage.includes('500')) {
      return 'Server error. Please try again later.';
    }

    // Default message
    return `Failed to ${operation}. Please try again.`;
  }

  /**
   * Check if error should offer retry option
   */
  private shouldOfferRetry(error: Error): boolean {
    const errorMessage = error.message.toLowerCase();
    const retryableErrors = ['network', 'fetch', 'timeout', 'server', '500'];
    return retryableErrors.some(keyword => errorMessage.includes(keyword));
  }

  /**
   * Handle retry logic
   */
  private handleRetry(context: ErrorContext): void {
    // Implement retry logic based on context
    console.log('Retrying operation:', context.operation);
  }

  /**
   * Save upload queue to AsyncStorage
   */
  private async saveUploadQueue(): Promise<void> {
    try {
      await AsyncStorage.setItem('upload_queue', JSON.stringify(this.uploadQueue));
    } catch (error) {
      console.error('Failed to save upload queue:', error);
    }
  }

  /**
   * Load upload queue from AsyncStorage
   */
  async loadUploadQueue(): Promise<void> {
    try {
      const queueData = await AsyncStorage.getItem('upload_queue');
      if (queueData) {
        this.uploadQueue = JSON.parse(queueData);
      }
    } catch (error) {
      console.error('Failed to load upload queue:', error);
    }
  }

  /**
   * Get authentication token
   */
  private async getAuthToken(): Promise<string> {
    // Implement your auth token retrieval logic
    return 'mock_token';
  }

  /**
   * Clear upload queue
   */
  async clearUploadQueue(): Promise<void> {
    this.uploadQueue = [];
    await AsyncStorage.removeItem('upload_queue');
  }

  /**
   * Get queue status
   */
  getQueueStatus(): { count: number; items: QueuedUpload[] } {
    return {
      count: this.uploadQueue.length,
      items: this.uploadQueue,
    };
  }
}

export default ErrorHandler.getInstance();
