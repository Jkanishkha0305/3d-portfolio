import {
    javascript,html,css,reactjs,tailwind,nodejs,mongodb,git,threejs,
    xneuronz,holopin,
    // nyu,bits,
  } from "../assets";

import nyu from "../assets/nyu.jpg"
import bits from "../assets/bits.jpg"
    
  const profiles = [
    // {
    //   link:"https://drive.google.com/file/d/1hAfQ2SFvW8XXFdgvrKo2w0Bm6hKNDPRw/view?usp=sharing",
    //   icon: "https://cdn.iconscout.com/icon/free/png-256/free-google-cloud-logo-icon-download-in-svg-png-gif-file-formats--weather-storage-data-pack-logos-icons-1721675.png?f=webp&w=256",
    // },
    {
      link: "https://drive.google.com/file/d/1qToSrv1H6W1YFCBqLGEpLE5Wi9HIo70C/view?usp=sharing",
      icon: "https://d3njjcbhbojbot.cloudfront.net/api/utilities/v1/imageproxy/http://coursera-university-assets.s3.amazonaws.com/b4/5cb90bb92f420b99bf323a0356f451/Icon.png?auto=format%2Ccompress&dpr=1&w=180&h=180",
    },
    {
      link: "https://drive.google.com/file/d/1ntkqytBIRZEXqgHdYQNZMGp1nE3WO55d/view?usp=sharing",
      icon: "https://cdn.icon-icons.com/icons2/2699/PNG/512/coursera_logo_icon_169326.png",
    },
    {
      link: "https://drive.google.com/file/d/1YZUAKlISMpajxgVp9BG4QAlyjjT0asmS/view?usp=sharing",
      icon: "https://identity.stanford.edu/wp-content/uploads/sites/3/2020/07/block-s-right.png",
    },
    {
      link: "https://drive.google.com/file/d/1Ie8h6SQCB01rqpMTPAhcZeSxWbhWZ_kK/view?usp=sharing",
      icon: "https://upload.wikimedia.org/wikipedia/en/thumb/0/04/Utoronto_coa.svg/250px-Utoronto_coa.svg.png",
    },
    {
      link:"https://drive.google.com/file/d/1SjB08sBkOdnVYhFSPnfVwaZvxJDHKTAR/view?usp=sharing",
      icon: "https://www.freepnglogos.com/uploads/ibm-logo-png/ibm-logo-mmtmagonline-benchurlm-urlscan-22.png",
    },
    {
      link:"https://drive.google.com/file/d/190uxeWvmM_IX0r2WzYTZ6nSd4hX1L9VZ/view?usp=sharing",
      icon: "https://cdn4.iconfinder.com/data/icons/logos-brands-7/512/google_logo-google_icongoogle-512.png",
    },
    {
      link:"https://forage-uploads-prod.s3.amazonaws.com/completion-certificates/British%20Airways/NjynCWzGSaWXQCxSX_British%20Airways_DKBmvcy5JJqJmzkFk_1701791416591_completion_certificate.pdf",
      icon: "https://logos-world.net/wp-content/uploads/2021/02/British-Airways-Symbol.png",
    },
    {
      link:"https://forage-uploads-prod.s3.amazonaws.com/completion-certificates/Cognizant/5N2ygyhzMWjKQmgCK_Cognizant_DKBmvcy5JJqJmzkFk_1696073922686_completion_certificate.pdf",
      icon: "https://logosandtypes.com/wp-content/uploads/2022/03/cognizant.svg",
    },
    {
      link:"https://forage-uploads-prod.s3.amazonaws.com/completion-certificates/BCG%20/Tcz8gTtprzAS4xSoK_BCG_DKBmvcy5JJqJmzkFk_1696079998294_completion_certificate.pdf",
      icon: "https://pbs.twimg.com/profile_images/1598063866285502464/OtjrY4Ly_400x400.png",
    },
  ];

  const achievements = [
    {
      title: "🏁 Finalist (Top 6) at Empire Hacks 2026, Cornell University.",
      link: "https://www.linkedin.com/posts/jkanishkha_empirehacks2026-cornelltech-aiagents-ugcPost-7442221055294898176-3p8g?utm_source=share&utm_medium=member_desktop&rcm=ACoAADXHBlUBj6-XUaYiIKHi1XNr9PdWjaOg5e0",
    },
    {
      title: "🚀 Top 5 at Build with AI Hackathon by GDG NYC during NYC Open Data Week 2026.",
      link: "https://www.linkedin.com/posts/jkanishkha_buildwithai-gdg-nycopendataweek-ugcPost-7445109603098357760-jp6c?utm_source=share&utm_medium=member_desktop&rcm=ACoAADXHBlUBj6-XUaYiIKHi1XNr9PdWjaOg5e0",
    },
    {
      title: "🥈 Second Place in Real-Time Data AI Hackathon (LENSES X AWS X Ness), NYC 2025.",
      link: "https://www.linkedin.com/posts/jkanishkha_ai-hackathon-streamingdata-ugcPost-7386317287085101056-fmNT?utm_source=share&utm_medium=member_desktop&rcm=ACoAADXHBlUBj6-XUaYiIKHi1XNr9PdWjaOg5e0",
    },
    {
      title: "🥇 First Place in Best use of Linkup, AI Agents Hackathon NYC, 2025.",
      link: "https://www.linkedin.com/posts/jkanishkha_ai-hackathonnyc-winners-ugcPost-7381192671794606080-F3Z3?utm_source=share&utm_medium=member_desktop&rcm=ACoAADXHBlUBj6-XUaYiIKHi1XNr9PdWjaOg5e0",
    },
    {
      title: "🥇 First Place in HackWallStreet x Deskhead Hackathon, 2024. ",
      link: "https://www.linkedin.com/posts/jkanishkha_hackstartup-innovation-financialdata-share-7249141490562564097-evXo?utm_source=share&utm_medium=member_desktop&rcm=ACoAADXHBlUBj6-XUaYiIKHi1XNr9PdWjaOg5e0",
    },
    {
      title: "🥉 Third Place in Hatch Labs Hackathon (Team Great Village), 2024 .",
      link: "https://www.linkedin.com/posts/jkanishkha_hackathon-ai-innovation-ugcPost-7264523027457949696-g3wY?utm_source=share&utm_medium=member_desktop&rcm=ACoAADXHBlUBj6-XUaYiIKHi1XNr9PdWjaOg5e0",
    },
    {
      title: "🥉 Third Place in HackBoston AI Hackathon, 2024.",
      link: "https://www.linkedin.com/posts/jkanishkha_hackbostonai-ai-thankful-share-7262524825988968448-1qOn?utm_source=share&utm_medium=member_desktop&rcm=ACoAADXHBlUBj6-XUaYiIKHi1XNr9PdWjaOg5e0",
    },
    {
      title: "🥇 Winner of DOTLAS Datathon, 2023.",
    },
    {
      title: "🥇 Winner of Dubai World Police Summit 2023 AI Challenge, UAE",
    },
    {
      title: "🥇 Winner of PIED CEL’s Desert Hack Hackathon 2022.",
    },
    {
      title: "🥇 Winner of ACM BPDC’s Capture The Flag (CTF) Competition 2020.",
    },
    // {
    //   title: "🥈 Runner-up of MTC BPDC’s CodeBlitz Competition 2020.",
    // },
    {
      title: "🎖️ Shortlisted for Top 10 Finalists of the Emirates Robotics Competition 2023, by Dubai Future Foundation x RIT Dubai, UAE.",
    },
    {
      title: "🎖️ Shortlisted for top 20 finalists all over Arab nations in the ITAS Arab Youth Competition 2023, Qatar.",
    },
    {
      title: "🥇 Winners of BITS Sports Festival Cricket Tournament 2019.",
    },
    {
      title: "🥈 Runner up of BITS Sports Festival Cricket Tournament 2022.",
    },
    
  ]

  const research = [
    {
      title: "GPUTOK: GPU Accelerated Byte Level BPE Tokenization",
      status: "arXiv",
      venue: "arXiv · Mar 2026",
      year: "2026",
      summary:
        "Built a GPU-accelerated byte-level BPE tokenizer that follows GPT-2 merge rules, matches CPU tokenization, and runs about 1.7x faster than tiktoken and 7.6x faster than Hugging Face GPT-2 on the longest WikiText103 inputs.",
      link: "https://arxiv.org/abs/2603.02597",
      linkLabel: "Show publication",
      tags: ["CUDA", "Tokenization", "Prof Advisor: Prof. Mohammed Zahran"],
    },
    {
      title: "Align2Act: Instruction-Tuned Models for Human-Aligned Autonomous Driving",
      status: "Graduate Research",
      venue: "New York University · Center for Data Science",
      year: "2025",
      summary:
        "Investigated aligning autonomous driving agents with natural language instructions through instruction tuning and evaluation.",
      link: "https://arxiv.org/abs/2510.10503",
      linkLabel: "arxiv",
      tags: ["Prof Advisor: Dr. Mengye Ren"],
    },
    {
      title: "Population-Coded Spiking Neural Networks for High-Dimensional Robotic Control",
      status: "Graduate Research",
      venue: "New York University · Robotics & Neuromorphic Computing",
      year: "2025",
      summary:
        "Developed population-coded SNN architectures to enable efficient continuous control in robotics with neuromorphic hardware.",
      link: "https://arxiv.org/abs/2510.10516",
      linkLabel: "arxiv",
      tags: ["Prof Advisor: Dr. Todd Gureckis"],
    },
    {
      title: "Multi-Turn RL for On-Device Mental Health Agents",
      status: "In Progress",
      venue: "New York University Langone Health",
      summary:
        "Explored hierarchical reinforcement learning (ArCHer framework) to train small language models for adaptive, multi-turn therapeutic dialogue and efficient on-device deployment.",
      tags: ["Prof. Advisor: Dr. Zhe Chen"]
    },
    {
      title: "Classification of Microstructure Images of Metals Using Transfer Learning",
      status: "Published Paper",
      venue: "MDIS 2022 · Sibiu, Romania · Springer",
      year: "2022",
      summary:
        "Applied transfer learning to accurately classify microscopic metal structures, contributing to automated materials analysis.",
      link: "https://doi.org/10.1007/978-3-031-27034-5",
      linkLabel: "Springer DOI",
      tags: ["Computer Vision", "Transfer Learning", "Prof Advisor: Dr. Angel Arul Jothi"],
    },
    {
      title: "An End-to-End Hybrid Learning Model for Detection of Covid-19 from Chest X-ray Images",
      status: "Published Paper",
      venue: "ICAIA-ATCON-1 2023 · Bengaluru, India · IEEE",
      year: "2023",
      summary:
        "Designed a hybrid CNN + LSTM architecture to improve Covid-19 diagnosis from chest radiography with rigorous benchmarking.",
      link: "https://doi.org/10.1109/ICAIA57370.2023.10169832",
      linkLabel: "IEEE DOI",
      tags: ["Medical Imaging", "Hybrid Deep Learning", "Prof Advisor: Dr. Pranav Pawar"],
    },
    {
      title:
        "Multi-model Approach for Autonomous Driving: A State-of-the-Art Deep Learning Strategy on Traffic Sign, Vehicle, and Lane Detection",
      status: "arXiv",
      venue: "arXiv · Autonomous Driving",
      year: "2026",
      summary:
        "Explored an ensemble of perception models for autonomous driving, combining detection, segmentation, and tracking pipelines.",
      link: "https://arxiv.org/abs/2603.09255",
      linkLabel: "Show publication",
      tags: ["Autonomous Driving", "Deep Learning", "Prof Advisor: Dr. Pranav Pawar"],
    },
    
    
  ]
  
  
  const technologies = [
    {
      name: "Python",
      icon: "/tech-logos/python.svg",
    },
    {
      name: "PyTorch",
      icon: "/tech-logos/pytorch.svg",
    },
    {
      name: "TensorFlow",
      icon: "https://img.icons8.com/?size=100&id=n3QRpDA7KZ7P&format=png&color=000000",
    },
    {
      name: "FastAPI",
      icon: "https://cdn.simpleicons.org/fastapi/009688",
    },
    {
      name:"AWS",
      icon: "https://raw.githubusercontent.com/devicons/devicon/master/icons/amazonwebservices/amazonwebservices-original-wordmark.svg",
    },
    {
      name: "Google Cloud",
      icon: "https://cdn.simpleicons.org/googlecloud/4285F4",
    },
    {
      name: "Docker",
      icon: "https://img.icons8.com/color/480/docker.png",
    },
    {
      name: "Kubernetes",
      icon: "https://cdn.simpleicons.org/kubernetes/326CE5",
    },
    {
      name: "PostgreSQL",
      icon: "https://cdn.simpleicons.org/postgresql/4169E1",
    },
    {
      name: "MongoDB",
      icon: "https://cdn.simpleicons.org/mongodb/47A248",
    },
    {
      name: "Neo4j",
      icon: "https://cdn.simpleicons.org/neo4j/4581C3",
    },
    {
      name: "CUDA",
      icon: "https://cdn.simpleicons.org/nvidia/76B900",
    },
    {
      name: "Hugging Face",
      icon: "/tech-logos/huggingface.svg",
    },
    {
      name: "LangChain",
      icon: "https://lancedb.github.io/lancedb/assets/langchain.png",
    },
    {
      name: "Qdrant",
      icon: "https://cdn.simpleicons.org/qdrant/DC244C",
    },
    {
      name: "Next.js",
      icon: "https://cdn.simpleicons.org/nextdotjs/000000",
    },
    {
      name: "Git",
      icon: git,
    },
  ];

  const list = [
    {
      id: "HACK",
      title: "Hackathons",
    },
    {
      id: "SYS",
      title: "AI Systems",
    },
    {
      id: "RES",
      title: "Research",
    },
  ];

  export const llmProject = [
    {
      name: "CityNerve",
      description:
        "Built a multimodal civic-response platform that triages city incidents from photos and voice, then routes dispatch decisions with a live command center.",
      tags: [
        {
          name: "Gemini 2.0",
          color: "blue-text-gradient",
        },
        {
          name: "GCP Cloud Run",
          color: "green-text-gradient",
        },
        {
          name: "Firestore",
          color: "pink-text-gradient",
        },
      ],
      image: "https://raw.githubusercontent.com/Jkanishkha0305/citynerve/main/assets/command-center.png",
      source_link: "https://smart311-frontend-446616000971.us-east1.run.app/dashboard",
      source_code_link: "https://github.com/Jkanishkha0305/citynerve",
    },
    {
      name: "Parallax",
      description:
        "Built an agentic UX-audit system that converts messy user complaints into multi-persona website analysis and auto-files actionable GitHub issues.",
      tags: [
        {
          name: "Claude Agents",
          color: "blue-text-gradient",
        },
        {
          name: "Playwright",
          color: "green-text-gradient",
        },
        {
          name: "Next.js",
          color: "pink-text-gradient",
        },
      ],
      image: "https://raw.githubusercontent.com/Jkanishkha0305/parallax/main/assets/gen_cover.png",
      source_link: "https://parallax-ten-rho.vercel.app",
      source_code_link: "https://github.com/Jkanishkha0305/parallax",
    },
    {
      name: "GhostOps",
      description:
        "Built a transparent desktop AI overlay that sees the screen, executes computer-use tasks, and records/replays workflows using multi-agent routing.",
      tags: [
        {
          name: "DigitalOcean Gradient",
          color: "blue-text-gradient",
        },
        {
          name: "Electron",
          color: "green-text-gradient",
        },
        {
          name: "FastAPI",
          color: "pink-text-gradient",
        },
      ],
      image: "https://raw.githubusercontent.com/Jkanishkha0305/ghostops/main/assets/GhostOps.png",
      source_link: "https://clownfish-app-dqd9h.ondigitalocean.app/health",
      source_code_link: "https://github.com/Jkanishkha0305/ghostops",
    },
    {
      name: "FlagSplainer",
      description:
        "Built a real-time fraud intelligence pipeline with Kafka and LangChain agents that flags suspicious transactions and explains each alert in plain language.",
      tags: [
        {
          name: "Kafka",
          color: "blue-text-gradient",
        },
        {
          name: "LangChain",
          color: "green-text-gradient",
        },
        {
          name: "AWS + Lenses",
          color: "pink-text-gradient",
        },
      ],
      image: "/project-covers/flagsplainer.png",
      source_link: "https://github.com/Jkanishkha0305/FlagSplainer",
      source_code_link: "https://github.com/Jkanishkha0305/FlagSplainer",
    },
    {
      name: "BillFighter",
      description:
        "Built a healthcare billing copilot that extracts line items from hospital bills, benchmarks CPT costs to CMS rates, and drafts dispute-ready letters.",
      tags: [
        {
          name: "Vertex AI",
          color: "blue-text-gradient",
        },
        {
          name: "Google ADK",
          color: "green-text-gradient",
        },
        {
          name: "FastAPI + Next.js",
          color: "pink-text-gradient",
        },
      ],
      image: "https://raw.githubusercontent.com/Jkanishkha0305/billfighter/main/assets/01-homepage.png",
      source_link: "https://github.com/Jkanishkha0305/billfighter",
      source_code_link: "https://github.com/Jkanishkha0305/billfighter",
    },
    {
      name: "StockPulse",
      description:
        "Built an AI-driven retail inventory system that combines demand signals and vendor negotiation to prevent stockouts and optimize purchase orders.",
      tags: [
        {
          name: "Airia Agent",
          color: "blue-text-gradient",
        },
        {
          name: "Supabase",
          color: "green-text-gradient",
        },
        {
          name: "Lightdash",
          color: "pink-text-gradient",
        },
      ],
      image: "/project-covers/stockpulse2.png",
      source_link: "https://github.com/Jkanishkha0305/StockPulse",
      source_code_link: "https://github.com/Jkanishkha0305/StockPulse",
    },
  ];
  
  export const cProject = [
    {
      name: "MCP PR Reviewer",
      description:
        "Built an MCP-native automation stack that turns GitHub PR events into cross-system review summaries posted into Slack with enterprise tool orchestration.",
      tags: [
        {
          name: "MCP",
          color: "blue-text-gradient",
        },
        {
          name: "Gemini",
          color: "green-text-gradient",
        },
        {
          name: "GitHub + Slack",
          color: "pink-text-gradient",
        },
      ],
      image: "/project-covers/mcp.png",
      source_link: "https://github.com/Jkanishkha0305/mcp-pr-reviewer",
      source_code_link: "https://github.com/Jkanishkha0305/mcp-pr-reviewer",
    },
    {
      name: "Argus",
      description:
        "Built a multimodal video understanding system that combines transcript, vision, and retrieval signals to answer grounded questions over long-form video.",
      tags: [
        {
          name: "FastMCP",
          color: "blue-text-gradient",
        },
        {
          name: "Pixeltable",
          color: "green-text-gradient",
        },
        {
          name: "FastAPI + React",
          color: "pink-text-gradient",
        },
      ],
      image: "/project-covers/argus.png",
      source_link: "https://github.com/Jkanishkha0305/Argus",
      source_code_link: "https://github.com/Jkanishkha0305/Argus",
    },
    {
      name: "PulseStream",
      description:
        "Built a real-time ICU monitoring pipeline with tiered anomaly detection to surface clinical risk from streaming vitals with low-latency alerts.",
      tags: [
        {
          name: "FastAPI",
          color: "blue-text-gradient",
        },
        {
          name: "Numba",
          color: "green-text-gradient",
        },
        {
          name: "Next.js + Supabase",
          color: "pink-text-gradient",
        },
      ],
      image: "/project-covers/pulsestream1.jpg",
      source_link: "https://github.com/Jkanishkha0305/PulseStream",
      source_code_link: "https://github.com/Jkanishkha0305/PulseStream",
    },
    {
      name: "ArcFlux",
      description:
        "Built an AI payment automation system that interprets natural language commands to schedule and execute USDC payments with risk checks.",
      tags: [
        {
          name: "FastAPI",
          color: "blue-text-gradient",
        },
        {
          name: "OpenAI Agents",
          color: "green-text-gradient",
        },
        {
          name: "Circle API",
          color: "pink-text-gradient",
        },
      ],
      image: "/project-covers/arcflux.png",
      source_link: "https://github.com/Jkanishkha0305/ArcFlux",
      source_code_link: "https://github.com/Jkanishkha0305/ArcFlux",
    },
    {
      name: "ClinicalChat Backend",
      description:
        "Built the backend service layer for clinical conversational workflows, handling APIs, orchestration, and production-ready healthcare data operations.",
      tags: [
        {
          name: "FastAPI",
          color: "blue-text-gradient",
        },
        {
          name: "PostgreSQL",
          color: "green-text-gradient",
        },
        {
          name: "REST APIs",
          color: "pink-text-gradient",
        },
      ],
      image: "/project-covers/axentra_landing.png",
      source_link: "https://github.com/Jkanishkha0305/clinicalchatbackend",
      source_code_link: "https://github.com/Jkanishkha0305/clinicalchatbackend",
    },
    {
      name: "MediBot GraphRAG",
      description:
        "Built a graph-RAG medical assistant over hospital entities and visit data using Neo4j and LangChain for structured clinical retrieval.",
      tags: [
        {
          name: "Neo4j",
          color: "blue-text-gradient",
        },
        {
          name: "LangChain",
          color: "green-text-gradient",
        },
        {
          name: "Docker",
          color: "pink-text-gradient",
        },
      ],
      image: "/project-covers/medibot.png",
      source_link: "https://github.com/Jkanishkha0305/Medibot-GraphRAG",
      source_code_link: "https://github.com/Jkanishkha0305/Medibot-GraphRAG",
    },
  ];

  export const webProject = [
    {
      name: "ReFine-Lab",
      description:
        "Developed a rubric-verifiable GRPO framework to behaviorally align small mental-health language models using clinically grounded rewards.",
      tags: [
        {
          name: "GRPO",
          color: "blue-text-gradient",
        },
        {
          name: "TRL",
          color: "green-text-gradient",
        },
        {
          name: "PEFT",
          color: "pink-text-gradient",
        },
      ],
      image: "/project-covers/refinelab.png",
      source_link: "https://github.com/Jkanishkha0305/ReFine-Lab",
      source_code_link: "https://github.com/Jkanishkha0305/ReFine-Lab",
    },
    {
      name: "Align2Act",
      description:
        "Built instruction-tuned planning for autonomous driving that maps scene understanding to human-aligned driving trajectories in simulation.",
      tags: [
        {
          name: "PyTorch",
          color: "blue-text-gradient",
        },
        {
          name: "LLaMA2",
          color: "green-text-gradient",
        },
        {
          name: "nuPlan",
          color: "pink-text-gradient",
        },
      ],
      image: "https://raw.githubusercontent.com/Jkanishkha0305/Align2Act/main/assets/viz.gif",
      source_link: "https://github.com/Jkanishkha0305/Align2Act",
      source_code_link: "https://github.com/Jkanishkha0305/Align2Act",
    },
    {
      name: "GPUTOK",
      description:
        "Built a GPU byte-level BPE tokenizer matching GPT-2 merge behavior and accelerating long-context tokenization against CPU baselines.",
      tags: [
        {
          name: "CUDA",
          color: "blue-text-gradient",
        },
        {
          name: "BlockBPE",
          color: "green-text-gradient",
        },
        {
          name: "Tokenization",
          color: "pink-text-gradient",
        },
      ],
      image: "/project-covers/gputok.png",
      source_link: "https://arxiv.org/abs/2603.02597",
      source_code_link: "https://github.com/Jkanishkha0305/gpu-tokenizer",
    },
    {
      name: "Theramind",
      description:
        "Built a privacy-first on-device mental wellness companion with local LLM inference, persistent chat memory, and polished mobile UX.",
      tags: [
        {
          name: "React Native",
          color: "blue-text-gradient",
        },
        {
          name: "llama.cpp",
          color: "green-text-gradient",
        },
        {
          name: "GGUF",
          color: "pink-text-gradient",
        },
      ],
      image: "/project-covers/theramind.png",
      source_link: "https://www.dropbox.com/scl/fi/wylnfdej17svr0hjzjcdp/Theramind_v1.MP4?rlkey=35grs9eyzro4p7jzc1u4qw72a&st=y392fmwq&dl=0",
      source_code_link: "https://github.com/Jkanishkha0305/Theramind",
    },
    {
      name: "LLM Serving at Scale",
      description:
        "Built a production-focused LLM serving stack with vLLM, Kubernetes autoscaling, and quantization to improve throughput and memory efficiency.",
      tags: [
        {
          name: "vLLM",
          color: "blue-text-gradient",
        },
        {
          name: "Kubernetes",
          color: "green-text-gradient",
        },
        {
          name: "W4A16",
          color: "pink-text-gradient",
        },
      ],
      image: "/project-covers/llmserving.png",
      source_link: "https://github.com/Jkanishkha0305/llm-serving-at-scale",
      source_code_link: "https://github.com/Jkanishkha0305/llm-serving-at-scale",
    },
    {
      name: "FactRL",
      description:
        "Built an RL-based fact-checking agent pipeline that trains tool-use behavior with GRPO to improve evidence-grounded claim verification.",
      tags: [
        {
          name: "GRPO",
          color: "blue-text-gradient",
        },
        {
          name: "QLoRA",
          color: "green-text-gradient",
        },
        {
          name: "Mistral",
          color: "pink-text-gradient",
        },
      ],
      image: "/project-covers/factrl.png",
      source_link: "https://github.com/Jkanishkha0305/FactRL",
      source_code_link: "https://github.com/Jkanishkha0305/FactRL",
    },
  ];

  export const otherProject = [];
  
  const experiences = [
    {
      title: "AI Engineer Intern",
      company_name: "Toyota Motors North America",
      icon: "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ee/Toyota_logo_%28Red%29.svg/960px-Toyota_logo_%28Red%29.svg.png",
      iconBg: "#ffffff",
      date: "Feb 2026 - Present",
      link: "https://www.toyota.com",
      points: [
        "Built a DeepAgents pipeline with supervisor, parser, scorer, and publisher subagents to convert 8K+ Toyota knowledge-base chunks across 900+ documents into reusable agentic skills, improving RAG performance and reducing unnecessary LLM calls by 3.5x.",
      ],
    },
    {
      title: "Graduate Student Researcher",
      company_name: "NYU Langone Health",
      icon: "https://pbs.twimg.com/profile_images/1675847757850988544/qDE0W0XQ_400x400.jpg",
      iconBg: "#570f8c",
      date: "Mar 2025 - Present",
      link: "https://nyulangone.org",
      points: [
        "Developed RV-GRPO, a rubric-verifiable reinforcement learning framework for therapeutic dialogue using clinically grounded rewards to replace subjective LLM-judge signals.",
      ],
    },
    {
      title: "Machine Learning Engineer",
      company_name: "Genmab",
      icon: "https://download.logo.wine/logo/Genmab/Genmab-Logo.wine.png",
      iconBg: "#ffffff",
      date: "Sep 2025 - Dec 2025",
      link: "https://www.genmab.com",
      points: [
        "Architected a clinical trials discovery engine using FastAPI, MongoDB with 500k+ records, and Qdrant hybrid search, enabling sub-second responses and 50 to 70% faster trial discovery.",
      ],
    },
    {
      title: "Machine Learning Engineer Intern",
      company_name: "Open Healthcare US",
      icon: "https://media.licdn.com/dms/image/v2/D4E0BAQHRXPhuM8m0jw/company-logo_200_200/B4EZmXdjPzKgAI-/0/1759182730641/open_healthcare_us_logo?e=2147483647&v=beta&t=5g8g1b3Rhqzw598Gauj6HQOJ8jPNvVQHmpx-RwuKLAM",
      iconBg: "#fcfcfcff",
      date: "May 2025 - Aug 2025",
      link: "https://us.ohc.global/",
      points: [
        "Developed a full-stack conversational lead-generation system integrating OpenAI, Apollo.io, Salesforce, and FastAPI, reducing manual sales operations by 40%.",
      ],
    },
    {
      title: "Founding Engineer",
      company_name: "LaRa Home (Stealth Mode)",
      icon: "https://media.licdn.com/dms/image/v2/D4D22AQGvPp1yHZduLQ/feedshare-shrink_800/B4DZdXwMkWGUAk-/0/1749523943164?e=2147483647&v=beta&t=hyasPp_V_-4Jffecec29aKanJ1ohi3ojNPsuE9d5I-U",
      iconBg: "#232452",
      date: "Feb 2025 - Dec 2025",
      link: "",
      points: [
        "Built a 0 to 1 AI compliance platform with React, FastAPI, Supabase, and Gemini API for architectural floorplan review, reducing manual compliance review time by 65%.",
      ],
    },
    {
      title: "Machine Learning Engineer",
      company_name: "Formetry Labs Pvt Ltd",
      icon: "https://media.licdn.com/dms/image/v2/D560BAQGP7fo1r0ky6Q/company-logo_200_200/company-logo_200_200/0/1707857824526/xneuronz_ai_logo?e=2147483647&v=beta&t=ZM_WDVGDQzQL7W41uE_pdPhkSoEex-Ep5ZexaosjSko",
      iconBg: "#000000",
      date: "Feb 2024 - Sep 2024",
      link: "",
      points: [
        "Engineered production RAG pipelines with Airflow, PostgreSQL, OpenSearch, Jina embeddings, and LangChain for 9000+ page construction documents, reducing manual lookup time by 60%.",
      ],
    },
    {
      title: "Data Scientist",
      company_name: "Dotlas",
      icon: "https://avatars.githubusercontent.com/u/88832003?s=200&v=4",
      iconBg: "#0c152c",
      date: "Sep 2023 - Jan 2024",
      link: "https://www.dotlas.com",
      points: [
        "Built data products for restaurant and housing intelligence using sentiment analysis, semi-supervised learning, SQL, and data visualization workflows.",
      ],
    },
    {
      title: "Data Science Intern",
      company_name: "KPTAC Technologies",
      icon: "https://drive.google.com/thumbnail?id=1A_mhCe4CDa-htvQsVOjTmqE6MxJTbP3q&sz=w1000",
      iconBg: "#f3343c",
      date: "Feb 2023 - Jul 2023",
      link: "",
      points: [
        "Built large-scale web data collection and geospatial analysis pipelines that improved grocery data acquisition and delivery optimization.",
      ],
    },
    // {
    //   title: "Undergraduate Student Researcher",
    //   company_name: "Birla Institute of Technology and Science, Pilani",
    //   icon: "https://upload.wikimedia.org/wikipedia/en/d/d3/BITS_Pilani-Logo.svg",
    //   iconBg: "#E6DEDD",
    //   date: "Jun 2022 - Jan 2023",
    //   link: "https://www.bits-pilani.ac.in/dubai/",
    //   points: [
    //     'Worked under guidance of Dr. Pranav M. Pawar to develop a cutting edge Hybrid Learning Model using a CNN+LSTM approach, achieving 97.56% accuracy in COVID-19 detection from chest X-ray images.(Published)',
    //     "Worked under Dr. Angel Arul Jothi on comparative analysis of Fine-tuned Transfer Learning Models for classifying microstructure images of metals using a custom-created dataset.(Published)",
    //     "Worked under supervisor Dr. Kalaiselvi on Application of DCNN for Visual Tracking of Mobile Robot using UAV.(MS2 - Mechatronics Lab)",
    //   ],
    //   link: "https://drive.google.com/file/d/1ztAkzEFMw97jih_T_Puwaw8guAVabyjk/view?usp=sharing",
    // },
    {
      title: "Software Engineering Intern",
      company_name: "Sentient Labs",
      icon: "https://static.wixstatic.com/media/973f4e_0d76a042752149ddae62463b656d746c~mv2.png",
      iconBg: "#E6DEDD",
      date: "Jun 2021 - Sep 2021",
      link: "https://www.sentientlabs.tech",
      points: [
        "Developed ROS-based robotics applications in AWS RoboMaker for simulated navigation, obstacle avoidance, and path planning.",
      ],
    },
    // {
    //   title: "Hacktoberfest",
    //   company_name: "GitHub",
    //   icon: hf,
    //   iconBg: "#E6DEDD",
    //   date: "Oct (2020 - 2022)",
    //   link: "",
    //   points: [
    //     "Hacktoberfest is an annual worldwide event held during the month of October. The event encourages open source developers to contribute to repositories through pull requests (PR).",
    //     "GitHub hosts many open source repositories that contribute to this event.",
    //   ],
    //   link: "https://dev.to/shinchancode",
    // },
  ];
  
  const educations = [
    {
      degree: "Masters of Science",
      branch:
        "Data Science",
      // marks:
      //   "CGPA : 9.42 / 10",
      name: "New York University",
      year: "2024 - 2026",
      image: "https://www.ssrc.org/wp-content/uploads/2022/08/NYU-modified.png",
    },
    {
      degree:
        "Bachelor of Engineering",
      branch : "Computer Science",
      // marks:
      //   "Percentage : 89.88 %",
      name: "Birla Institute of Technology and Science, Pilani",
      year: "2019 - 2023",
      image: "https://drive.google.com/thumbnail?id=1S0I3tjM7o9cvvCTZYkoyPIjG__7nB4PA&sz=w1000",
    },
    // {
    //   degree:
    //     "10th Grade",
    //   branch: "SSC",
    //   marks:
    //     "Percentage : 95 %",
    //   name: "Kendriya Vidyalaya RHE Khadki Pune",
    //   year: "2016",
    //   image: school,
    // },
  ];

  const about = {
    name: "Kanishkha",
    title: "Machine Learning Engineer",
    description1:
      "I am a Machine Learning Engineer focused on building AI systems that move cleanly from research to production.",
    description2:
      "My experience spans NYU Center for Data Science, Toyota, Genmab, and early-stage startups across applied AI and platform work.",
    description3:
      "I work on multi-agent workflows, LLM applications, and practical ML products in healthcare and enterprise settings.",
    description4:
      "I care about clear problem framing, measurable outcomes, and engineering that stays reliable outside the notebook.",
  };
  
  export { list, profiles, technologies, experiences, educations, achievements, research, about };
