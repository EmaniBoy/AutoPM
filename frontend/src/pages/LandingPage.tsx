import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Send, Paperclip, X, ExternalLink, ArrowRight } from 'lucide-react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:5001/api';

export default function LandingPage() {
  console.log('🏠 LandingPage component is rendering...');
  
  const [input, setInput] = useState('');
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [showChatbot, setShowChatbot] = useState(false);
  const [agentTypes, setAgentTypes] = useState<('jira' | 'github' | 'research')[]>([]);
  const [userMessage, setUserMessage] = useState('');
  const [isThinking, setIsThinking] = useState(false);
  const [showResponse, setShowResponse] = useState(false);
  const [orchestratorData, setOrchestratorData] = useState<any>(null);
  const navigate = useNavigate();

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files);
      setSelectedFiles(files);
      
      // Upload each file immediately
      for (const file of files) {
        const formData = new FormData();
        formData.append('file', file);
        
        try {
          const response = await axios.post(`${API_BASE_URL}/upload-file`, formData, {
            withCredentials: false,
            headers: { 'Content-Type': 'multipart/form-data' }
          });
          console.log('File uploaded:', response.data);
        } catch (error) {
          console.error('Error uploading file:', error);
        }
      }
    }
  };

  const removeFile = (index: number) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() || selectedFiles.length > 0) {
      const prompt = input.trim();
      
      if (!prompt) {
        return;
      }
      
      setUserMessage(prompt);
      setIsThinking(true);
      setInput('');
      setSelectedFiles([]);
      
      try {
        console.log('🚀 Submitting prompt:', prompt);
        console.log('🔗 API URL:', `${API_BASE_URL}/run-orchestrator`);
        
        // Run orchestrator
        const response = await axios.post(`${API_BASE_URL}/run-orchestrator`, {
          prompt: prompt
        }, {
          withCredentials: false,
          headers: { 'Content-Type': 'application/json' }
        });
        
        const data = response.data;
        setOrchestratorData(data);
        
        const intent = data.intent || '';
        const summary = data.summary || '';
        const hasMetrics = data.has_metrics || false;
        
        console.log('API Response:', { intent, summary: summary.substring(0, 100), hasMetrics });
        
        // Determine routing based on API intent (prioritize API response over keyword matching)
        const inputLower = prompt.toLowerCase();
        
        // Check intent from API first
        const intentLower = intent.toLowerCase();
        const hasJiraIntent = intentLower.includes('jira');
        const hasDevopsIntent = intentLower.includes('devops');
        const hasResearchIntent = intentLower.includes('research') && !hasJiraIntent && !hasDevopsIntent;
        
        // Check if prompt asks for all 3 agents
        const asksForAllAgents = inputLower.includes('all') && (
          (inputLower.includes('agent') || inputLower.includes('ai')) &&
          (inputLower.includes('devops') || inputLower.includes('github') || inputLower.includes('sprint') || inputLower.includes('jira')) &&
          (inputLower.includes('research') || inputLower.includes('analyze') || inputLower.includes('findings'))
        );
        
        // Fallback to keyword matching only if intent is unclear
        const hasJira = hasJiraIntent || (!intent && (inputLower.includes('jira') || inputLower.includes('sprint') || inputLower.includes('epic') || inputLower.includes('story')));
        const hasGithub = hasDevopsIntent || (!intent && (inputLower.includes('github') || inputLower.includes('devops') || inputLower.includes('ci/cd') || inputLower.includes('workflow')));
        const hasResearch = hasResearchIntent || (!intent && (
          inputLower.includes('research') || 
          (inputLower.includes('analyze') && inputLower.includes('banking_corpus')) ||
          (inputLower.includes('extract') && (inputLower.includes('problem') || inputLower.includes('finding')))
        ));
        
        console.log('Routing decision:', { hasJira, hasGithub, hasResearch, intent, asksForAllAgents });
        
        // If asking for all 3 agents, show all of them
        if (asksForAllAgents || (hasJira && hasGithub && hasResearch)) {
          const agents: ('jira' | 'github' | 'research')[] = ['jira', 'github', 'research'];
          setAgentTypes(agents);
          setShowChatbot(true);
          
          setTimeout(() => {
            setIsThinking(false);
            setShowResponse(true);
            
            // Play notification sound
            const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
            const playNotification = () => {
              const oscillator = audioContext.createOscillator();
              const gainNode = audioContext.createGain();
              oscillator.connect(gainNode);
              gainNode.connect(audioContext.destination);
              oscillator.frequency.setValueAtTime(800, audioContext.currentTime);
              oscillator.frequency.setValueAtTime(1000, audioContext.currentTime + 0.1);
              gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
              gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
              oscillator.start(audioContext.currentTime);
              oscillator.stop(audioContext.currentTime + 0.5);
            };
            playNotification();
          }, 2000);
        } else if (hasJira || hasGithub) {
          // Route to chatbot with agent types
          const agents: ('jira' | 'github' | 'research')[] = [];
          if (hasJira) agents.push('jira');
          if (hasGithub) agents.push('github');
          if (hasResearch && (hasJira || hasGithub)) agents.push('research');
          
          setAgentTypes(agents);
          setShowChatbot(true);
          
          setTimeout(() => {
            setIsThinking(false);
            setShowResponse(true);
            
            // Play notification sound
            const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
            const playNotification = () => {
              const oscillator = audioContext.createOscillator();
              const gainNode = audioContext.createGain();
              oscillator.connect(gainNode);
              gainNode.connect(audioContext.destination);
              oscillator.frequency.setValueAtTime(800, audioContext.currentTime);
              oscillator.frequency.setValueAtTime(1000, audioContext.currentTime + 0.1);
              gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
              gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
              oscillator.start(audioContext.currentTime);
              oscillator.stop(audioContext.currentTime + 0.5);
            };
            playNotification();
          }, 2000);
        } else if (hasResearch) {
          // Route to ModelSelection page with data
          navigate('/models', { 
            state: { 
              input: prompt,
              summary: summary,
              hasMetrics: hasMetrics
            } 
          });
        } else {
          // Default: go to loading then models
          navigate('/loading', { state: { input: prompt } });
        }
      } catch (error) {
        console.error('Error running orchestrator:', error);
        setIsThinking(false);
        // Fallback to loading screen
        navigate('/loading', { state: { input: prompt } });
      }
    }
  };

  const getAgentName = (type: 'jira' | 'github' | 'research') => {
    if (type === 'jira') return 'Sprint Agent';
    if (type === 'github') return 'DevOps Agent';
    return 'Research Agent';
  };

  if (showChatbot && agentTypes.length > 0) {
    return (
      <div className="min-h-screen bg-neutral-950 relative overflow-hidden">
        <div className="absolute inset-0 pointer-events-none">
          <div
            className="w-[600px] h-[600px] bg-emerald-500/10 rounded-full blur-[120px] transition-transform duration-200 ease-out"
            style={{
              transform: `translate(${mousePosition.x - 300}px, ${mousePosition.y - 300}px)`
            }}
          />
        </div>

        <div className="relative z-10 h-screen flex flex-col">
          <div className="flex-1 overflow-y-auto p-8">
            <div className="max-w-3xl mx-auto space-y-4 pb-6">
              <div className="flex justify-end opacity-0 translate-x-8 animate-[slideInRight_0.5s_ease-out_forwards]">
                <div className="bg-emerald-600 text-white px-6 py-3 rounded-3xl max-w-lg">
                  {userMessage}
                </div>
              </div>

              {isThinking && agentTypes.map((type, index) => (
                <div
                  key={type}
                  className="flex justify-start opacity-0 -translate-x-8 animate-[slideInLeft_0.5s_ease-out_forwards]"
                  style={{ animationDelay: `${0.3 + index * 0.2}s` }}
                >
                  <div className="bg-neutral-900/50 border border-neutral-800 px-6 py-3 rounded-3xl">
                    <div className="flex items-center gap-2 text-neutral-400">
                      <div className="flex gap-1">
                        <span className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                        <span className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                        <span className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                      </div>
                      <span>{getAgentName(type)} is thinking...</span>
                    </div>
                  </div>
                </div>
              ))}

              {showResponse && agentTypes.map((type, index) => {
                if (type === 'research') {
                  // Research agent - show summary and metrics below DevOps and Sprint
                  const researchIndex = agentTypes.indexOf('research');
                  const devopsIndex = agentTypes.indexOf('github');
                  const sprintIndex = agentTypes.indexOf('jira');
                  
                  // Only show research after DevOps and Sprint have been shown
                  if (researchIndex > devopsIndex && researchIndex > sprintIndex) {
                    return (
                      <div
                        key={type}
                        className="flex justify-start opacity-0 -translate-x-8 animate-[slideInLeft_0.6s_ease-out_forwards]"
                        style={{ animationDelay: `${index * 0.3}s` }}
                      >
                        <div className="bg-neutral-900/50 border border-neutral-800 rounded-3xl p-6 max-w-4xl w-full">
                          <div className="text-emerald-400 text-lg font-medium mb-4">
                            Research Agent
                          </div>
                          
                          {/* Summary Section */}
                          {orchestratorData?.summary && (
                            <div className="mb-6">
                              <h4 className="text-neutral-100 font-semibold mb-3 text-base">
                                Research Summary
                              </h4>
                              <div className="bg-neutral-800/50 rounded-2xl p-5 border border-neutral-700/50">
                                <div className="text-neutral-300 text-sm leading-relaxed whitespace-pre-wrap">
                                  {orchestratorData.summary}
                                </div>
                              </div>
                            </div>
                          )}

                          {/* Metrics Chart Section */}
                          {orchestratorData?.has_metrics && (
                            <div className="mb-6">
                              <h4 className="text-neutral-100 font-semibold mb-3 text-base">
                                Data Visualization
                              </h4>
                              <div className="bg-neutral-800/50 rounded-2xl p-5 border border-neutral-700/50">
                                <div className="bg-neutral-900/70 rounded-xl p-4 flex items-center justify-center min-h-[300px]">
                                  <img
                                    src={`${API_BASE_URL}/metrics-chart?t=${Date.now()}`}
                                    alt="Metrics visualization"
                                    className="max-w-full h-auto rounded-lg"
                                    onError={(e) => {
                                      e.currentTarget.style.display = 'none';
                                      const fallback = e.currentTarget.nextElementSibling as HTMLElement;
                                      if (fallback) fallback.classList.remove('hidden');
                                    }}
                                  />
                                  <div className="hidden text-neutral-500 text-center">
                                    <svg className="w-16 h-16 mx-auto mb-2 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
                                    </svg>
                                    <p className="text-sm">Data visualization placeholder</p>
                                  </div>
                                </div>
                              </div>
                            </div>
                          )}

                          {/* Key Findings / String Data */}
                          {orchestratorData?.summary && (
                            <div className="mt-6 pt-4 border-t border-neutral-700/50">
                              <h4 className="text-neutral-100 font-semibold mb-3 text-base">
                                Key Findings
                              </h4>
                              <div className="text-neutral-300 text-sm space-y-2">
                                {/* Parse summary for key findings if it contains bullet points or numbered items */}
                                {orchestratorData.summary.split('\n').filter((line: string) => 
                                  line.trim().startsWith('-') || 
                                  line.trim().match(/^\d+\./) ||
                                  line.trim().startsWith('•')
                                ).map((line: string, idx: number) => (
                                  <div key={idx} className="bg-neutral-800/30 rounded-lg p-3">
                                    {line.trim()}
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  }
                  return null;
                }

                return (
                  <div
                    key={type}
                    className="flex justify-start opacity-0 -translate-x-8 animate-[slideInLeft_0.6s_ease-out_forwards]"
                    style={{ animationDelay: `${index * 0.3}s` }}
                  >
                    <div className="bg-neutral-900/50 border border-neutral-800 rounded-3xl p-6 max-w-lg">
                      <div className="text-emerald-400 text-lg font-medium mb-3">
                        {getAgentName(type)}
                      </div>
                      <p className="text-neutral-300 text-lg mb-4">
                        has successfully completed your request
                      </p>
                      <a
                        href={type === 'jira'
                          ? orchestratorData?.links?.jira_url || 'https://id.atlassian.com/login?continue=https%3A%2F%2Fwww.atlassian.com%2Fgateway%2Fapi%2Fstart%2Fauthredirect%3Fcontinue%3Dhttps%3A%2F%2Fwww.atlassian.com%2Fsoftware%2Fjira%2Fproduct-discovery%3Fcampaign%3D22198605308%26adgroup%3D176115566084%26targetid%3Dkwd-904779413018%26matchtype%3De%26network%3Dg%26device%3Dc%26device_model%3D%26creative%3D752512191326%26keyword%3Dproduct%2520management%2520jira%26placement%3D%26target%3D%26ds_e1%3DGOOGLE%26gad_source%3D1%26gad_campaignid%3D22198605308%26gbraid%3D0AAAAA-uFwoKJHfvV1yMQJfDQ8XY5GGrVP%26gclid%3DCjwKCAiA8bvIBhBJEiwAu5ayrO5WHjXzHovTl-BdX_RHexKUiii-vEobBmqM-KpMqF7IoGJJ0Ga7BBoC3ooQAvD_BwE'
                          : orchestratorData?.links?.pr_url || orchestratorData?.links?.github_url || 'https://github.com/login'
                        }
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-2 bg-emerald-600 hover:bg-emerald-500 text-white px-6 py-2 rounded-full transition-all duration-300 font-medium hover:scale-105 hover:shadow-lg hover:shadow-emerald-500/20"
                      >
                        Go to {type === 'jira' ? 'Jira' : 'GitHub'}
                        <ExternalLink className="w-4 h-4" />
                      </a>
                    </div>
                  </div>
                );
              })}

              {showResponse && (
                <div
                  className="flex justify-start opacity-0 animate-[slideInLeft_0.5s_ease-out_forwards]"
                  style={{ animationDelay: `${agentTypes.length * 0.3 + 0.2}s` }}
                >
                  <button
                    onClick={() => {
                      setShowChatbot(false);
                      setAgentTypes([]);
                      setUserMessage('');
                      setIsThinking(false);
                      setShowResponse(false);
                    }}
                    className="text-neutral-400 hover:text-emerald-400 transition-all duration-300"
                  >
                    Back to home
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-neutral-950 flex items-center justify-center p-4 relative overflow-hidden">
      <div className="absolute inset-0 pointer-events-none">
        <div
          className="w-[600px] h-[600px] bg-emerald-500/10 rounded-full blur-[120px] transition-transform duration-200 ease-out"
          style={{
            transform: `translate(${mousePosition.x - 300}px, ${mousePosition.y - 300}px)`
          }}
        />
      </div>

      <div className="max-w-3xl w-full z-10">
        <h1 className="text-center text-4xl md:text-5xl font-semibold text-neutral-100 mb-8">
          Your Workflow. Automated.
        </h1>

        <form onSubmit={handleSubmit} className="relative mb-4">
          <div className="relative">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="What are your project management needs?"
              className="w-full rounded-full bg-black/40 border border-neutral-800/80 text-neutral-200 px-6 py-4 pr-16 text-lg placeholder:text-neutral-500 focus:outline-none focus:ring-1 focus:ring-emerald-500/40 focus:border-emerald-500/60 transition"
            />
            <button
              type="submit"
              className="absolute right-2 top-1/2 -translate-y-1/2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-full p-3 transition"
            >
              <Send className="w-5 h-5" />
            </button>
          </div>

          <div className="flex items-center gap-3 mt-3 px-2">
            <label className="flex items-center gap-2 text-neutral-400 hover:text-emerald-500 cursor-pointer transition text-sm">
              <Paperclip className="w-4 h-4" />
              <span>Add context (PDF/Image/TXT)</span>
              <input
                type="file"
                accept=".pdf,.jpg,.jpeg,.png,.txt"
                multiple
                onChange={handleFileChange}
                className="hidden"
              />
            </label>
          </div>

          {selectedFiles.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-2 px-2">
              {selectedFiles.map((file, index) => (
                <div
                  key={index}
                  className="flex items-center gap-2 bg-neutral-900/50 border border-neutral-800 rounded-full px-3 py-1 text-sm text-neutral-300"
                >
                  <span className="truncate max-w-[200px]">{file.name}</span>
                  <button
                    type="button"
                    onClick={() => removeFile(index)}
                    className="text-neutral-500 hover:text-emerald-500 transition"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </form>

        <div className="flex gap-3 justify-center flex-wrap">
          <button
            onClick={() => window.open('https://id.atlassian.com/login?continue=https%3A%2F%2Fwww.atlassian.com%2Fgateway%2Fapi%2Fstart%2Fauthredirect%3Fcontinue%3Dhttps%3A%2F%2Fwww.atlassian.com%2Fsoftware%2Fjira%2Fproduct-discovery%3Fcampaign%3D22198605308%26adgroup%3D176115566084%26targetid%3Dkwd-904779413018%26matchtype%3De%26network%3Dg%26device%3Dc%26device_model%3D%26creative%3D752512191326%26keyword%3Dproduct%2520management%2520jira%26placement%3D%26target%3D%26ds_e1%3DGOOGLE%26gad_source%3D1%26gad_campaignid%3D22198605308%26gbraid%3D0AAAAA-uFwoKJHfvV1yMQJfDQ8XY5GGrVP%26gclid%3DCjwKCAiA8bvIBhBJEiwAu5ayrO5WHjXzHovTl-BdX_RHexKUiii-vEobBmqM-KpMqF7IoGJJ0Ga7BBoC3ooQAvD_BwE', '_blank')}
            className="rounded-full px-5 py-2 bg-neutral-900 border border-neutral-700 hover:bg-neutral-800 text-neutral-100 transition flex items-center gap-2"
          >
            Connect Jira
            <ExternalLink className="w-4 h-4" />
          </button>
          <button
            onClick={() => window.open('https://github.com/login', '_blank')}
            className="rounded-full px-5 py-2 bg-neutral-900 border border-neutral-700 hover:bg-neutral-800 text-neutral-100 transition flex items-center gap-2"
          >
            Connect GitHub
            <ExternalLink className="w-4 h-4" />
          </button>
          <button
            onClick={() => navigate('/pitch-deck')}
            className="rounded-full px-5 py-2 bg-neutral-900 border border-neutral-700 hover:bg-neutral-800 text-neutral-100 transition"
          >
            View Pitch Deck
          </button>
        </div>
      </div>
    </div>
  );
}
