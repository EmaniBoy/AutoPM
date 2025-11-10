import { useNavigate, useLocation, useParams } from 'react-router-dom';
import { Sparkles, ArrowLeft, ArrowRight, CheckCircle } from 'lucide-react';

export default function ModelDetail() {
  const navigate = useNavigate();
  const location = useLocation();
  const { modelId } = useParams();
  const userInput = location.state?.input || 'Your project';

  const modelData: Record<string, any> = {
    research: {
      name: 'Research Agent',
      icon: Sparkles,
      color: 'emerald',
      achievements: [
        {
          title: 'Market Analysis',
          description: 'Conducted comprehensive market research identifying key competitors and opportunities in the project management space.',
        },
        {
          title: 'User Requirements',
          description: 'Gathered and analyzed user needs from 50+ potential customers, identifying pain points in current Workflows.',
        },
        {
          title: 'Technology Stack',
          description: 'Researched and recommended optimal technology stack including React, Node.js, and modern AI frameworks.',
        },
        {
          title: 'Competitive Analysis',
          description: 'Created detailed competitive landscape analysis with SWOT assessment of top 5 competitors.',
        },
        {
          title: 'Feasibility Study',
          description: 'Validated technical and business feasibility with risk assessment and mitigation strategies.',
        },
      ],
    },
  };

  const model = modelData[modelId || ''];

  if (!model) {
    return (
      <div className="min-h-screen bg-neutral-950 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl text-neutral-100 mb-4">Model not found</h1>
          <button
            onClick={() => navigate('/models', { state: { input: userInput } })}
            className="px-6 py-3 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl transition"
          >
            Back to Models
          </button>
        </div>
      </div>
    );
  }

  const Icon = model.icon;

  return (
    <div className="min-h-screen bg-neutral-950 p-8">
      <div className="max-w-5xl mx-auto">
        <div className="flex justify-between items-start mb-12">
          <button
            onClick={() => navigate('/models', { state: { input: userInput } })}
            className="px-5 py-2.5 bg-neutral-900 hover:bg-neutral-800 text-neutral-100 rounded-xl transition flex items-center justify-center gap-2 border border-neutral-800"
          >
            <ArrowLeft className="w-5 h-5" />
            Back to Models
          </button>
          <button
            onClick={() => navigate('/pitch-deck', { state: { input: userInput } })}
            className="px-6 py-3 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl transition flex items-center gap-2 font-medium"
          >
            View Pitch Deck
            <ArrowRight className="w-5 h-5" />
          </button>
        </div>

        <div className="bg-neutral-900 border-2 border-neutral-800 rounded-3xl p-8 mb-8">
          <div className="flex items-center gap-6 mb-4">
            <div className="w-20 h-20 rounded-2xl bg-neutral-800 flex items-center justify-center">
              <Icon className="w-10 h-10 text-emerald-400" />
            </div>
            <div>
              <h1 className="text-3xl md:text-4xl font-semibold text-neutral-100 mb-2">
                {model.name}
              </h1>
              <p className="text-neutral-400">
                For: <span className="text-emerald-400">{userInput}</span>
              </p>
            </div>
          </div>
        </div>

        <h2 className="text-2xl font-semibold text-neutral-100 mb-6">Key Achievements</h2>

        <div className="space-y-4">
          {model.achievements.map((achievement: any, index: number) => (
            <div
              key={index}
              className="bg-neutral-900 border border-neutral-800 rounded-2xl p-6 hover:border-emerald-500/30 transition-colors"
            >
              <div className="flex items-start gap-4">
                <div className="mt-1">
                  <CheckCircle className="w-6 h-6 text-emerald-400" />
                </div>
                <div className="flex-1">
                  <h3 className="text-xl font-semibold text-neutral-100 mb-2">
                    {achievement.title}
                  </h3>
                  <p className="text-neutral-400 leading-relaxed">
                    {achievement.description}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
