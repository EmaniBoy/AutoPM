import { useNavigate, useLocation } from 'react-router-dom';
import { ArrowRight, BarChart3 } from 'lucide-react';
import { useState, useEffect } from 'react';

const API_BASE_URL = 'http://localhost:5001/api';

export default function ModelSelection() {
  const navigate = useNavigate();
  const location = useLocation();
  const userInput = location.state?.input || 'Your project';
  const summaryFromState = location.state?.summary || '';
  const hasMetricsFromState = location.state?.hasMetrics || false;
  
  const [summary] = useState(summaryFromState);
  const [metricsChartUrl, setMetricsChartUrl] = useState<string | null>(null);

  useEffect(() => {
    // Load metrics chart if available
    if (hasMetricsFromState) {
      setMetricsChartUrl(`${API_BASE_URL}/metrics-chart?t=${Date.now()}`);
    }
  }, [hasMetricsFromState]);

  return (
    <div className="min-h-screen bg-neutral-950 p-8">
      <div className="max-w-6xl mx-auto">
        <div className="flex justify-between items-start mb-8">
          <div>
            <h1 className="text-3xl md:text-4xl font-semibold text-neutral-100 mb-2">
              Research Agent Results
            </h1>
            <p className="text-neutral-600">
              Automation and results from RAG model
            </p>
            <p className="text-neutral-400 mt-2">
              Your Prompt: <span className="text-emerald-400">{userInput}</span>
            </p>
          </div>
          <button
            onClick={() => navigate('/pitch-deck', { state: { input: userInput } })}
            className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg transition flex items-center gap-2 text-sm font-medium"
          >
            View Pitch Deck
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>

        {/* Summary Section */}
        <div className="bg-neutral-900 border border-neutral-800 rounded-2xl p-8 mb-8">
          <h2 className="text-2xl font-semibold text-neutral-100 mb-4">Summary</h2>
          <div className="text-neutral-300 leading-relaxed whitespace-pre-wrap">
            {summary || 'No summary available'}
          </div>
        </div>

        {/* Data Visualization Section */}
        <div className="bg-neutral-900 border border-neutral-800 rounded-2xl p-8">
          <h2 className="text-2xl font-semibold text-neutral-100 mb-4">Data & Statistics</h2>
          <div className="bg-neutral-800 rounded-xl p-6 min-h-[400px] flex items-center justify-center">
            {metricsChartUrl ? (
              <img
                src={metricsChartUrl}
                alt="Metrics visualization"
                className="max-w-full h-auto rounded-lg"
                onError={(e) => {
                  e.currentTarget.style.display = 'none';
                  e.currentTarget.nextElementSibling!.classList.remove('hidden');
                }}
              />
            ) : null}
            <div className={`text-center text-neutral-400 ${metricsChartUrl ? 'hidden' : ''}`}>
              <BarChart3 className="w-16 h-16 mx-auto mb-4 text-emerald-400" />
              <p className="text-lg mb-2">Data Visualization</p>
              <p className="text-sm">Chart or graph image can be displayed here</p>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
