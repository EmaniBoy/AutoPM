import { useNavigate, useLocation } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';

export default function PitchDeck() {
  const navigate = useNavigate();
  const location = useLocation();
  const userInput = location.state?.input || 'Your project';

  return (
    <div className="min-h-screen bg-neutral-950 p-8">
      <div className="flex justify-between items-center mb-8">
        <button
          onClick={() => navigate('/models', { state: { input: userInput } })}
          className="flex items-center justify-center gap-2 text-neutral-400 hover:text-emerald-500 transition"
        >
          <ArrowLeft className="w-5 h-5" />
          Back to Research Agent Results
        </button>
        <button
          onClick={() => navigate('/')}
          className="px-5 py-2.5 bg-neutral-900 hover:bg-neutral-800 text-neutral-100 rounded-xl transition border border-neutral-800 text-center"
        >
          Home
        </button>
      </div>

      <div className="max-w-5xl mx-auto">
        <h1 className="text-3xl md:text-4xl font-semibold text-neutral-100 mb-12 text-center">
          Your Pitch Deck
        </h1>

        <div className="bg-neutral-900 border-2 border-emerald-500/30 rounded-2xl p-8">
          <h2 className="text-2xl font-semibold text-emerald-400 mb-4 text-center">Pitch Deck</h2>
          <div className="text-neutral-300 space-y-2">
          </div>
        </div>
      </div>
    </div>
  );
}
