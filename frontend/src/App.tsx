import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import LandingPage from './pages/LandingPage';
import LoadingScreen from './pages/LoadingScreen';
import ModelSelection from './pages/ModelSelection';
import ModelDetail from './pages/ModelDetail';
import PitchDeck from './pages/PitchDeck';

console.log('📱 App component is loading...');

function App() {
  console.log('📱 App component is rendering...');
  
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/loading" element={<LoadingScreen />} />
        <Route path="/models" element={<ModelSelection />} />
        <Route path="/model/:modelId" element={<ModelDetail />} />
        <Route path="/pitch-deck" element={<PitchDeck />} />
      </Routes>
    </Router>
  );
}

export default App;
