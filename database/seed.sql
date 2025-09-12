-- Khet Guard Database Seed Data

-- Insert sample users
INSERT INTO users (firebase_uid, name, email, phone, region, language) VALUES
('firebase_uid_rajesh', 'Farmer Rajesh Kumar', 'rajesh@example.com', '+919812345678', 'Maharashtra', 'hi'),
('firebase_uid_anita', 'Farmer Anita Singh', 'anita@example.com', '+919876543210', 'Punjab', 'en'),
('firebase_uid_ramesh', 'Farmer Ramesh Patel', 'ramesh@example.com', '+919876543211', 'Gujarat', 'hi');

-- Insert pesticide recommendations
INSERT INTO pesticide_recommendations (disease_name, recommended, dosage, safety, organic_alternatives) VALUES
('Tomato___Early_blight', ARRAY['Mancozeb 75% WP', 'Chlorothalonil 75% WP'], '2g per litre of water, spray weekly', 'Wear gloves and mask while spraying. Avoid contact with skin and eyes.', ARRAY['Neem oil', 'Copper fungicide', 'Baking soda solution']),
('Tomato___Late_blight', ARRAY['Metalaxyl 8% + Mancozeb 64% WP', 'Copper oxychloride 50% WP'], '2.5g per litre, repeat every 10 days', 'Avoid spraying close to harvest. Do not exceed 3 applications per season.', ARRAY['Copper fungicide', 'Bordeaux mixture', 'Garlic extract']),
('Tomato___Leaf_Mold', ARRAY['Azoxystrobin 23% SC', 'Chlorothalonil 75% WP'], '1.5g per litre, apply at first sign of disease', 'Do not exceed 3 sprays per season. Maintain proper ventilation.', ARRAY['Neem oil', 'Sulfur dust', 'Baking soda spray']),
('Tomato___Bacterial_spot', ARRAY['Copper oxychloride 50% WP', 'Streptomycin sulfate'], '2g per litre, spray every 7-10 days', 'Use protective equipment. Avoid spraying during flowering.', ARRAY['Copper fungicide', 'Garlic extract', 'Horsetail tea']),
('Tomato___healthy', ARRAY['Consult local agricultural expert'], 'N/A', 'Get proper diagnosis before treatment', ARRAY['Preventive measures', 'Good cultural practices', 'Regular monitoring']),
('Corn_(maize)___Common_rust_', ARRAY['Propiconazole 25% EC', 'Tebuconazole 25% EC'], '1ml per litre, spray at first sign', 'Apply during early morning or evening. Avoid during flowering.', ARRAY['Neem oil', 'Sulfur dust', 'Copper fungicide']),
('Potato___Late_blight', ARRAY['Metalaxyl 8% + Mancozeb 64% WP'], '2.5g per litre, repeat every 10 days', 'Critical disease - apply preventively. Avoid overhead irrigation.', ARRAY['Copper fungicide', 'Bordeaux mixture', 'Garlic extract']);

-- Insert cattle breeds
INSERT INTO cattle_breeds (breed_name, origin, milk_yield, characteristics, feeding_requirements, vaccination_schedule) VALUES
('Gir', 'Gujarat, India', 'Medium to High (8-12 liters/day)', 'Hardy, disease resistant, known for A2 milk, good for crossbreeding', 'Green fodder, concentrate feed, mineral supplements', 'Annual vaccination for FMD, HS, BQ. Deworming every 3 months'),
('Sahiwal', 'Punjab region, India/Pakistan', 'High (10-15 liters/day)', 'Best dairy breed, heat tolerant, good milk quality', 'High quality green fodder, balanced concentrate, adequate water', 'Regular vaccination schedule. Heat stress management important'),
('Kangayam', 'Tamil Nadu, India', 'Low to Medium (3-5 liters/day)', 'Primarily draught breed, very hardy, drought resistant', 'Local grass, minimal concentrate, salt licks', 'Basic vaccination program. Focus on draught animal care'),
('Tharparkar', 'Rajasthan, India', 'Medium (6-8 liters/day)', 'Dual purpose: milk and draught, drought resistant, good temperament', 'Mixed feeding system, drought-resistant fodder', 'Standard vaccination. Special care during drought periods'),
('Holstein_Friesian', 'Netherlands', 'Very High (15-25 liters/day)', 'High milk production, requires good management', 'High quality feed, balanced nutrition, adequate water', 'Intensive vaccination program. Regular health monitoring'),
('Jersey', 'Jersey Island', 'High (12-18 liters/day)', 'Efficient milk producer, good butterfat content', 'Quality green fodder, concentrate feed, mineral supplements', 'Regular vaccination. Heat stress management'),
('Murrah', 'Haryana, India', 'High (8-12 liters/day)', 'Best buffalo breed, high fat content in milk', 'Green fodder, concentrate, adequate water supply', 'Standard buffalo vaccination program'),
('Nili_Ravi', 'Punjab, India/Pakistan', 'High (7-10 liters/day)', 'Good milk producer, docile temperament', 'Balanced feeding, good quality fodder', 'Regular vaccination and health checkups');

-- Insert some scan history
INSERT INTO scans (user_id, image_url, prediction_type, class_name, confidence, class_id, grad_cam_url, pesticide_recommendation, breed_info, metadata)
SELECT user_id, 
       'https://storage.googleapis.com/khet-guard-images/leaf1.jpg', 
       'disease_pest', 
       'Tomato___Early_blight', 
       0.945, 
       12, 
       'https://storage.googleapis.com/khet-guard-images/gradcam1.png',
       '{"recommended": ["Mancozeb 75% WP"], "dosage": "2g per litre", "safety": "Wear protective equipment"}',
       NULL,
       '{"location": {"lat": 19.0760, "lon": 72.8777}, "device": "Android", "app_version": "1.0.0"}'
FROM users WHERE email='rajesh@example.com';

INSERT INTO scans (user_id, image_url, prediction_type, class_name, confidence, class_id, grad_cam_url, pesticide_recommendation, breed_info, metadata)
SELECT user_id, 
       'https://storage.googleapis.com/khet-guard-images/cow1.jpg', 
       'cattle', 
       'Gir', 
       0.873, 
       9, 
       'https://storage.googleapis.com/khet-guard-images/gradcam2.png',
       NULL,
       '{"origin": "Gujarat, India", "milk_yield": "Medium to High", "characteristics": "Hardy, disease resistant"}',
       '{"location": {"lat": 31.1471, "lon": 75.3412}, "device": "iOS", "app_version": "1.0.0"}'
FROM users WHERE email='anita@example.com';

INSERT INTO scans (user_id, image_url, prediction_type, class_name, confidence, class_id, grad_cam_url, pesticide_recommendation, breed_info, metadata)
SELECT user_id, 
       'https://storage.googleapis.com/khet-guard-images/leaf2.jpg', 
       'disease_pest', 
       'Tomato___healthy', 
       0.912, 
       30, 
       'https://storage.googleapis.com/khet-guard-images/gradcam3.png',
       '{"recommended": ["Consult local agricultural expert"], "dosage": "N/A", "safety": "Get proper diagnosis before treatment"}',
       NULL,
       '{"location": {"lat": 23.0225, "lon": 72.5714}, "device": "Android", "app_version": "1.0.0"}'
FROM users WHERE email='ramesh@example.com';

-- Insert analytics data
INSERT INTO analytics (region, disease_name, scan_count, avg_confidence) VALUES
('Maharashtra', 'Tomato___Early_blight', 25, 0.89),
('Maharashtra', 'Tomato___healthy', 15, 0.92),
('Punjab', 'Sahiwal', 10, 0.87),
('Punjab', 'Gir', 8, 0.85),
('Tamil Nadu', 'Kangayam', 8, 0.78),
('Gujarat', 'Gir', 12, 0.91),
('Rajasthan', 'Tharparkar', 6, 0.83);

-- Insert user sessions
INSERT INTO user_sessions (user_id, device_info, app_version)
SELECT user_id, 
       '{"platform": "Android", "version": "12", "model": "Samsung Galaxy A52"}',
       '1.0.0'
FROM users WHERE email='rajesh@example.com';

INSERT INTO user_sessions (user_id, device_info, app_version)
SELECT user_id, 
       '{"platform": "iOS", "version": "15.7", "model": "iPhone 12"}',
       '1.0.0'
FROM users WHERE email='anita@example.com';

-- Insert some feedback
INSERT INTO feedback (scan_id, user_id, is_correct, actual_class, comments)
SELECT s.scan_id, s.user_id, true, 'Tomato___Early_blight', 'Prediction was accurate. The treatment worked well.'
FROM scans s 
JOIN users u ON s.user_id = u.user_id 
WHERE u.email='rajesh@example.com' AND s.class_name='Tomato___Early_blight'
LIMIT 1;
