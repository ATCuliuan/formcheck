import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, Modal, TextInput, FlatList } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';

// The master list of exercises your app supports
const AVAILABLE_EXERCISES = ['Squat', 'Push-up', 'Lunge', 'Plank', 'Deadlift', 'Bicep Curl', 'Jumping Jacks'];

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const [selectedExercise, setSelectedExercise] = useState('squat');
  
  // New States for the Searchable Dropdown Modal
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const [telemetry, setTelemetry] = useState({
    rep_count: 0,
    form_score: 100,
    feedback: 'Position full body in camera',
    state: 'READY'
  });
  
  const cameraRef = useRef<any>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const ws = new WebSocket(`ws://10.0.2.2:8000/ws/track/${selectedExercise.toLowerCase()}`);
    wsRef.current = ws;

    ws.onopen = () => console.log('Connected to Python Backend WebSocket');

    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.status === 'tracking') {
          setTelemetry({
            rep_count: data.rep_count,
            form_score: data.form_score,
            feedback: data.feedback,
            state: data.state
          });
        } else if (data.status === 'warning') {
          setTelemetry((prev) => ({ ...prev, feedback: data.feedback }));
        }
      } catch (err) {
        console.error('Error parsing backend telemetry', err);
      }
    };

    ws.onerror = (e) => console.log('WebSocket error:', e);
    ws.onclose = () => console.log('WebSocket closed');

    return () => {
      ws.close();
    };
  }, [selectedExercise]);

  useEffect(() => {
    const interval = setInterval(async () => {
      if (cameraRef.current && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        try {
          const photo = await cameraRef.current.takePictureAsync({
            base64: true,
            quality: 0.3,
            skipProcessing: true,
          });
          if (photo?.base64) {
            wsRef.current.send(photo.base64);
          }
        } catch (err) {}
      }
    }, 500);

    return () => clearInterval(interval);
  }, []);

  // Filter exercises based on search query
  const filteredExercises = AVAILABLE_EXERCISES.filter(ex => 
    ex.toLowerCase().includes(searchQuery.toLowerCase())
  );

  if (!permission) return <View style={styles.container} />;
  
  if (!permission.granted) {
    return (
      <View style={styles.centerContainer}>
        <Text style={styles.permissionText}>Camera permission is required.</Text>
        <TouchableOpacity style={styles.primaryButton} onPress={requestPermission}>
          <Text style={styles.buttonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.brandSubtitle}>FORMCHECK</Text>
        <Text style={styles.brandTitle}>
          FIX YOUR <Text style={styles.highlightText}>FORM.</Text>
        </Text>
      </View>

      {/* Main Camera Viewport */}
      <View style={styles.cameraCard}>
        <CameraView style={styles.camera} ref={cameraRef} facing="back" />

        <View style={styles.trackingBadge}>
          <Text style={styles.trackingLabel}>NOW TRACKING</Text>
          <Text style={styles.trackingExercise}>
            {selectedExercise.charAt(0).toUpperCase() + selectedExercise.slice(1)}
          </Text>
        </View>

        <View style={styles.feedbackBanner}>
          <View style={styles.statusDot} />
          <Text style={styles.feedbackText}>{telemetry.feedback}</Text>
        </View>

        <View style={styles.repCircle}>
          <Text style={styles.repNumber}>{telemetry.rep_count}</Text>
          <Text style={styles.repLabel}>REPS</Text>
        </View>
      </View>

      {/* NEW: Searchable Dropdown Button */}
      <TouchableOpacity 
        style={styles.dropdownSelector} 
        onPress={() => setIsModalVisible(true)}
      >
        <Text style={styles.dropdownSelectorLabel}>Current Exercise</Text>
        <View style={styles.dropdownRow}>
            <Text style={styles.dropdownSelectorText}>
            {selectedExercise.charAt(0).toUpperCase() + selectedExercise.slice(1)}
            </Text>
            <Text style={styles.dropdownIcon}>▼</Text>
        </View>
      </TouchableOpacity>

      {/* Stats Cards */}
      <View style={styles.statsRow}>
        <View style={styles.statCard}>
          <Text style={styles.statNumber}>{telemetry.rep_count}</Text>
          <Text style={styles.statLabel}>THIS SET</Text>
        </View>
        <View style={styles.statCard}>
          <Text style={styles.statNumber}>{telemetry.form_score}%</Text>
          <Text style={styles.statLabel}>FORM SCORE</Text>
        </View>
        <View style={styles.statCard}>
          <Text style={styles.statNumber}>{telemetry.state}</Text>
          <Text style={styles.statLabel}>STATE</Text>
        </View>
      </View>

      <TouchableOpacity style={styles.actionButton}>
        <Text style={styles.actionButtonText}>▶ Start tracking session</Text>
      </TouchableOpacity>

      {/* NEW: The Modal Popup for Searching Exercises */}
      <Modal visible={isModalVisible} animationType="slide" transparent={true}>
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Select Exercise</Text>
            
            <TextInput
              style={styles.searchInput}
              placeholder="Search (e.g. Squat)..."
              placeholderTextColor="#888"
              value={searchQuery}
              onChangeText={setSearchQuery}
            />

            <FlatList
              data={filteredExercises}
              keyExtractor={(item) => item}
              renderItem={({ item }) => (
                <TouchableOpacity 
                  style={styles.modalListItem} 
                  onPress={() => {
                    setSelectedExercise(item.toLowerCase());
                    setIsModalVisible(false);
                    setSearchQuery(''); // Reset search
                  }}
                >
                  <Text style={styles.modalListItemText}>{item}</Text>
                </TouchableOpacity>
              )}
              ListEmptyComponent={
                <Text style={styles.emptyListText}>No exercises found.</Text>
              }
            />

            <TouchableOpacity 
              style={styles.closeModalButton} 
              onPress={() => {
                setIsModalVisible(false);
                setSearchQuery('');
              }}
            >
              <Text style={styles.closeModalButtonText}>Cancel</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#EFECE6', paddingHorizontal: 20, paddingTop: 50 },
  centerContainer: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 20 },
  permissionText: { fontSize: 16, textAlign: 'center', marginBottom: 20, color: '#333' },
  primaryButton: { backgroundColor: '#E64A19', paddingVertical: 12, paddingHorizontal: 24, borderRadius: 25 },
  buttonText: { color: '#FFF', fontWeight: 'bold' },
  header: { marginTop: 10, marginBottom: 15 },
  brandSubtitle: { fontSize: 12, fontWeight: '700', color: '#888', letterSpacing: 1.5 },
  brandTitle: { fontSize: 26, fontWeight: '900', color: '#1A1A1A' },
  highlightText: { color: '#E64A19' },
  cameraCard: { height: 320, borderRadius: 24, overflow: 'hidden', backgroundColor: '#1E1E1E', marginBottom: 15, position: 'relative' },
  camera: { ...StyleSheet.absoluteFillObject },
  trackingBadge: { position: 'absolute', top: 15, left: 15, backgroundColor: 'rgba(0,0,0,0.6)', paddingVertical: 6, paddingHorizontal: 12, borderRadius: 12 },
  trackingLabel: { color: '#AAA', fontSize: 10, fontWeight: '700' },
  trackingExercise: { color: '#FFF', fontSize: 14, fontWeight: 'bold' },
  feedbackBanner: { position: 'absolute', bottom: 15, left: 15, right: 110, backgroundColor: 'rgba(255, 255, 255, 0.92)', paddingVertical: 10, paddingHorizontal: 14, borderRadius: 20, flexDirection: 'row', alignItems: 'center' },
  statusDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: '#2E7D32', marginRight: 8 },
  feedbackText: { color: '#1B5E20', fontSize: 13, fontWeight: '700' },
  repCircle: { position: 'absolute', bottom: 15, right: 15, width: 80, height: 80, borderRadius: 40, backgroundColor: 'rgba(0, 0, 0, 0.8)', borderWidth: 3, borderColor: '#E64A19', justifyContent: 'center', alignItems: 'center' },
  repNumber: { color: '#FFF', fontSize: 24, fontWeight: '900' },
  repLabel: { color: '#AAA', fontSize: 10, fontWeight: '700' },
  
  // New Dropdown Styles
  dropdownSelector: { backgroundColor: '#FFF', paddingVertical: 12, paddingHorizontal: 16, borderRadius: 16, marginBottom: 15, borderWidth: 1, borderColor: '#DDD' },
  dropdownSelectorLabel: { fontSize: 10, fontWeight: '700', color: '#888', marginBottom: 4 },
  dropdownRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  dropdownSelectorText: { fontSize: 18, fontWeight: '800', color: '#1A1A1A' },
  dropdownIcon: { fontSize: 16, color: '#1A1A1A' },

  statsRow: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 20 },
  statCard: { flex: 1, backgroundColor: '#FFF', paddingVertical: 14, marginHorizontal: 4, borderRadius: 16, alignItems: 'center', elevation: 1 },
  statNumber: { fontSize: 18, fontWeight: '800', color: '#1A1A1A' },
  statLabel: { fontSize: 10, fontWeight: '700', color: '#888', marginTop: 4 },
  actionButton: { backgroundColor: '#E64A19', paddingVertical: 16, borderRadius: 30, alignItems: 'center' },
  actionButtonText: { color: '#FFF', fontSize: 16, fontWeight: '800' },

  // New Modal Styles
  modalOverlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.5)', justifyContent: 'flex-end' },
  modalContent: { backgroundColor: '#FFF', borderTopLeftRadius: 24, borderTopRightRadius: 24, padding: 20, maxHeight: '80%' },
  modalTitle: { fontSize: 20, fontWeight: '800', color: '#1A1A1A', marginBottom: 15 },
  searchInput: { backgroundColor: '#F5F5F5', borderRadius: 12, padding: 12, fontSize: 16, marginBottom: 15, borderWidth: 1, borderColor: '#DDD' },
  modalListItem: { paddingVertical: 16, borderBottomWidth: 1, borderBottomColor: '#EEE' },
  modalListItemText: { fontSize: 16, fontWeight: '600', color: '#333' },
  emptyListText: { textAlign: 'center', color: '#888', marginTop: 20 },
  closeModalButton: { marginTop: 20, backgroundColor: '#1A1A1A', paddingVertical: 14, borderRadius: 16, alignItems: 'center' },
  closeModalButtonText: { color: '#FFF', fontSize: 16, fontWeight: '700' }
});