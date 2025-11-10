import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Hourglass } from 'lucide-react';

export default function LoadingScreen() {
  const navigate = useNavigate();
  const location = useLocation();
  const userInput = location.state?.input || 'Your project';

  const [progress, setProgress] = useState(0);
  const [dots, setDots] = useState('');

  useEffect(() => {
    const progressInterval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(progressInterval);
          return 100;
        }
        return prev + 2;
      });
    }, 100);

    const dotsInterval = setInterval(() => {
      setDots((prev) => {
        if (prev === '...') return '';
        return prev + '.';
      });
    }, 500);

    setTimeout(() => {
      navigate('/models', { state: { input: userInput } });
    }, 5000);

    return () => {
      clearInterval(progressInterval);
      clearInterval(dotsInterval);
    };
  }, [navigate, userInput]);

  return (
    <div className="min-h-screen bg-neutral-950 flex items-center justify-center p-8">
      <div className="max-w-2xl w-full text-center">
        <div className="mb-12 flex justify-center">
          <Hourglass
            className="w-24 h-24 text-emerald-400 animate-spin"
            style={{ animationDuration: '2s' }}
          />
        </div>

        <h1 className="text-2xl md:text-3xl font-medium text-neutral-100 mb-12">
          Automating your Workflow needs{dots}
        </h1>

        <div className="space-y-3">
          <div className="flex justify-between text-sm text-neutral-400 px-1">
            <span>Progress</span>
            <span>{Math.round(progress)}%</span>
          </div>
          <div className="w-full h-4 bg-neutral-900 rounded-full overflow-hidden border border-neutral-800">
            <div
              className="h-full bg-gradient-to-r from-emerald-600 to-emerald-400 transition-all duration-100 ease-linear"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
