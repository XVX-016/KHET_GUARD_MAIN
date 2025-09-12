/**
 * Image processing service for Khet Guard mobile app.
 * Handles image capture, preprocessing, validation, and compression.
 */

import * as ImagePicker from 'expo-image-picker';
import * as ImageManipulator from 'expo-image-manipulator';
import { Alert } from 'react-native';

export interface ImageProcessingOptions {
  quality: number;
  maxWidth: number;
  maxHeight: number;
  minWidth: number;
  minHeight: number;
  format: 'jpeg' | 'png';
}

export interface ProcessedImage {
  uri: string;
  base64: string;
  width: number;
  height: number;
  size: number;
  format: string;
}

export interface ImageValidationResult {
  isValid: boolean;
  errors: string[];
  warnings: string[];
}

class ImageProcessor {
  private static instance: ImageProcessor;
  private defaultOptions: ImageProcessingOptions = {
    quality: 0.7,
    maxWidth: 1024,
    maxHeight: 1024,
    minWidth: 256,
    minHeight: 256,
    format: 'jpeg',
  };

  static getInstance(): ImageProcessor {
    if (!ImageProcessor.instance) {
      ImageProcessor.instance = new ImageProcessor();
    }
    return ImageProcessor.instance;
  }

  /**
   * Request camera permissions
   */
  async requestCameraPermissions(): Promise<boolean> {
    try {
      const { status } = await ImagePicker.requestCameraPermissionsAsync();
      
      if (status !== 'granted') {
        Alert.alert(
          'Camera Permission Required',
          'Please enable camera access in your device settings to take photos for analysis.',
          [
            { text: 'Cancel', style: 'cancel' },
            { text: 'Settings', onPress: () => this.openSettings() },
          ]
        );
        return false;
      }
      
      return true;
    } catch (error) {
      console.error('Error requesting camera permissions:', error);
      return false;
    }
  }

  /**
   * Request media library permissions
   */
  async requestMediaLibraryPermissions(): Promise<boolean> {
    try {
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      
      if (status !== 'granted') {
        Alert.alert(
          'Photo Library Permission Required',
          'Please enable photo library access to select images for analysis.',
          [
            { text: 'Cancel', style: 'cancel' },
            { text: 'Settings', onPress: () => this.openSettings() },
          ]
        );
        return false;
      }
      
      return true;
    } catch (error) {
      console.error('Error requesting media library permissions:', error);
      return false;
    }
  }

  /**
   * Capture image from camera
   */
  async captureImage(options?: Partial<ImageProcessingOptions>): Promise<ProcessedImage | null> {
    try {
      // Check permissions
      const hasPermission = await this.requestCameraPermissions();
      if (!hasPermission) {
        return null;
      }

      // Launch camera
      const result = await ImagePicker.launchCameraAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [1, 1],
        quality: 0.8,
        base64: true,
      });

      if (result.canceled || !result.assets?.[0]) {
        return null;
      }

      const asset = result.assets[0];
      return await this.processImage(asset.uri, asset.base64, options);
    } catch (error) {
      console.error('Error capturing image:', error);
      Alert.alert('Error', 'Failed to capture image. Please try again.');
      return null;
    }
  }

  /**
   * Select image from gallery
   */
  async selectImage(options?: Partial<ImageProcessingOptions>): Promise<ProcessedImage | null> {
    try {
      // Check permissions
      const hasPermission = await this.requestMediaLibraryPermissions();
      if (!hasPermission) {
        return null;
      }

      // Launch image picker
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [1, 1],
        quality: 0.8,
        base64: true,
      });

      if (result.canceled || !result.assets?.[0]) {
        return null;
      }

      const asset = result.assets[0];
      return await this.processImage(asset.uri, asset.base64, options);
    } catch (error) {
      console.error('Error selecting image:', error);
      Alert.alert('Error', 'Failed to select image. Please try again.');
      return null;
    }
  }

  /**
   * Process and validate image
   */
  async processImage(
    uri: string, 
    base64: string | undefined, 
    options?: Partial<ImageProcessingOptions>
  ): Promise<ProcessedImage | null> {
    try {
      const opts = { ...this.defaultOptions, ...options };
      
      // Get image info
      const imageInfo = await ImageManipulator.manipulateAsync(
        uri,
        [],
        { format: ImageManipulator.SaveFormat.JPEG }
      );

      // Validate image
      const validation = this.validateImage(imageInfo, opts);
      if (!validation.isValid) {
        this.showValidationErrors(validation.errors);
        return null;
      }

      // Show warnings if any
      if (validation.warnings.length > 0) {
        this.showValidationWarnings(validation.warnings);
      }

      // Resize and compress image
      const processedImage = await ImageManipulator.manipulateAsync(
        uri,
        [
          {
            resize: {
              width: Math.min(imageInfo.width, opts.maxWidth),
              height: Math.min(imageInfo.height, opts.maxHeight),
            },
          },
        ],
        {
          compress: opts.quality,
          format: opts.format === 'jpeg' ? ImageManipulator.SaveFormat.JPEG : ImageManipulator.SaveFormat.PNG,
          base64: true,
        }
      );

      // Get base64 data
      const processedBase64 = processedImage.base64 || base64;
      if (!processedBase64) {
        throw new Error('Failed to get base64 data');
      }

      return {
        uri: processedImage.uri,
        base64: processedBase64,
        width: processedImage.width,
        height: processedImage.height,
        size: this.estimateImageSize(processedBase64),
        format: opts.format,
      };
    } catch (error) {
      console.error('Error processing image:', error);
      Alert.alert('Error', 'Failed to process image. Please try again.');
      return null;
    }
  }

  /**
   * Validate image quality and dimensions
   */
  validateImage(imageInfo: any, options: ImageProcessingOptions): ImageValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];

    // Check dimensions
    if (imageInfo.width < options.minWidth || imageInfo.height < options.minHeight) {
      errors.push(`Image too small. Minimum size: ${options.minWidth}x${options.minHeight}px`);
    }

    if (imageInfo.width > options.maxWidth || imageInfo.height > options.maxHeight) {
      warnings.push(`Image will be resized from ${imageInfo.width}x${imageInfo.height}px`);
    }

    // Check aspect ratio
    const aspectRatio = imageInfo.width / imageInfo.height;
    if (aspectRatio < 0.5 || aspectRatio > 2.0) {
      warnings.push('Image has unusual aspect ratio. Results may be less accurate.');
    }

    // Check if image is too dark or bright
    // This would require additional image analysis
    if (imageInfo.width * imageInfo.height < 50000) {
      warnings.push('Image resolution is low. Higher quality images give better results.');
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings,
    };
  }

  /**
   * Show validation errors to user
   */
  private showValidationErrors(errors: string[]): void {
    Alert.alert(
      'Image Quality Issue',
      errors.join('\n\n'),
      [{ text: 'OK' }]
    );
  }

  /**
   * Show validation warnings to user
   */
  private showValidationWarnings(warnings: string[]): void {
    Alert.alert(
      'Image Quality Warning',
      warnings.join('\n\n'),
      [{ text: 'OK' }]
    );
  }

  /**
   * Estimate image size in bytes
   */
  private estimateImageSize(base64: string): number {
    // Base64 encoding increases size by ~33%
    return Math.round((base64.length * 3) / 4);
  }

  /**
   * Open device settings
   */
  private openSettings(): void {
    // This would open device settings
    // Implementation depends on the platform
    console.log('Opening device settings...');
  }

  /**
   * Check if image is blurry (basic implementation)
   */
  async isImageBlurry(uri: string): Promise<boolean> {
    try {
      // This is a simplified blur detection
      // In production, you might want to use more sophisticated methods
      const imageInfo = await ImageManipulator.manipulateAsync(
        uri,
        [],
        { format: ImageManipulator.SaveFormat.JPEG }
      );

      // Simple heuristic: if image is very small, it might be blurry
      return imageInfo.width * imageInfo.height < 100000;
    } catch (error) {
      console.error('Error checking image blur:', error);
      return false;
    }
  }

  /**
   * Get image metadata
   */
  async getImageMetadata(uri: string): Promise<{
    width: number;
    height: number;
    format: string;
    size: number;
  }> {
    try {
      const imageInfo = await ImageManipulator.manipulateAsync(
        uri,
        [],
        { format: ImageManipulator.SaveFormat.JPEG }
      );

      return {
        width: imageInfo.width,
        height: imageInfo.height,
        format: 'jpeg',
        size: 0, // Would need to get actual file size
      };
    } catch (error) {
      console.error('Error getting image metadata:', error);
      throw error;
    }
  }

  /**
   * Create thumbnail for preview
   */
  async createThumbnail(uri: string, size: number = 150): Promise<string> {
    try {
      const thumbnail = await ImageManipulator.manipulateAsync(
        uri,
        [
          {
            resize: {
              width: size,
              height: size,
            },
          },
        ],
        {
          compress: 0.8,
          format: ImageManipulator.SaveFormat.JPEG,
        }
      );

      return thumbnail.uri;
    } catch (error) {
      console.error('Error creating thumbnail:', error);
      return uri; // Return original if thumbnail creation fails
    }
  }
}

export default ImageProcessor.getInstance();
