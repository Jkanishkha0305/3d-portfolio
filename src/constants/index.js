import {
    javascript,html,css,reactjs,tailwind,nodejs,mongodb,git,threejs,
    hf,xneuronz,holopin,
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
    },
    {
      title: "🚀 Top 5 at Build with AI Hackathon by GDG NYC during NYC Open Data Week 2026.",
    },
    {
      title: "🥈 Second Place in Real-Time Data AI Hackathon (LENSES X AWS X Ness), NYC 2025.",
    },
    {
      title: "🥇 First Place in Best use of Linkup, AI Agents Hackathon NYC, 2025.",
    },
    {
      title: "🥇 First Place in HackWallStreet x Deskhead Hackathon, 2024. ",
    },
    {
      title: "🥉 Third Place in Hatch Labs Hackathon (Team Great Village), 2024 .",
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
      tags: ["CUDA", "Tokenization", "Prof Advisor: Prof. Zahran"],
    },
    {
      title: "Align2Act: Instruction-Tuned Models for Human-Aligned Autonomous Driving",
      status: "Graduate Research",
      venue: "New York University · Center for Data Science",
      year: "2025",
      summary:
        "Investigates aligning autonomous driving agents with natural language instructions through instruction tuning and evaluation.",
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
        "Develops population-coded SNN architectures to enable efficient continuous control in robotics with neuromorphic hardware.",
      link: "https://arxiv.org/abs/2510.10516",
      linkLabel: "arxiv",
      tags: ["Prof Advisor: Dr. Todd Gureckis"],
    },
    {
      title: "Multi-Turn RL for On-Device Mental Health Agents",
      status: "In Progress",
      venue: "New York University Langone Health",
      summary:
        "Exploring hierarchical reinforcement learning (ArCHer framework) to train small language models for adaptive, multi-turn therapeutic dialogue and efficient on-device deployment.",
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
      tags: ["Computer Vision", "Transfer Learning"],
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
      tags: ["Medical Imaging", "Hybrid Deep Learning"],
    },
    {
      title:
        "Multi-model Approach for Autonomous Driving: A State-of-the-Art Deep Learning Strategy on Traffic Sign, Vehicle, and Lane Detection",
      status: "arXiv",
      venue: "arXiv · Autonomous Driving",
      year: "2026",
      summary:
        "Explores an ensemble of perception models for autonomous driving, combining detection, segmentation, and tracking pipelines.",
      link: "https://arxiv.org/abs/2603.09255",
      linkLabel: "Show publication",
      tags: ["Autonomous Driving", "Deep Learning"],
    },
    
    
  ]
  
  
  const technologies = [
    {
      name: "Python",
      icon: "https://img.icons8.com/?size=100&id=l75OEUJkPAk4&format=png&color=000000",
    },
    {
      name: "PyTorch",
      icon: "https://img.icons8.com/?size=100&id=jH4BpkMnRrU5&format=png&color=000000",
    },
    {
      name: "Tensorflow",
      icon: "https://img.icons8.com/?size=100&id=n3QRpDA7KZ7P&format=png&color=000000",
    },
    {
      name:"C++ tool",
      icon: "https://img.icons8.com/color/480/000000/c-plus-plus-logo.png",
    },
    {
      name:"R",
      icon: "https://img.icons8.com/?size=100&id=CLvQeiwFpit4&format=png&color=000000",
    },
    // {
    //   name: "scikit-learn",
    //   icon: "https://img.icons8.com/color/480/000000/scikit-learn.png",
    // },
    // {
    //   name: "Pandas",
    //   icon: "https://img.icons8.com/color/480/pandas.png",
    // },
    {
      name:"AWS",
      icon: "https://raw.githubusercontent.com/devicons/devicon/master/icons/amazonwebservices/amazonwebservices-original-wordmark.svg",
    },
    // {
    //   name: "MLflow",
    //   icon: "https://mlflow.org/docs/latest/_static/MLflow-logo-final-black.png",
    // },
    {
      name: "Docker",
      icon: "https://img.icons8.com/color/480/docker.png",
    },
    {
      name: "OpenCV",
      icon: "https://img.icons8.com/?size=100&id=bpip0gGiBLT1&format=png&color=000000",
    },
    {
      name: "Hugging Face",
      icon: "https://img.icons8.com/?size=100&id=sop9ROXku5bb&format=png&color=000000",
    },
    {
      name: "LlamaIndex",
      icon: "https://asset.brandfetch.io/id6a4s3gXI/idncpUsO_z.jpeg",
    },
    {
      name: "Langchain",
      icon: "https://lancedb.github.io/lancedb/assets/langchain.png",
    },
    // {
    //   name: "DSPy",
    //   icon: "https://dspy-docs.vercel.app/img/logo.png",
    // },
    {
      name: "git",
      icon: git,
    },
    {
      name:"MySql",
      icon: "https://img.icons8.com/color/480/000000/mysql-logo.png",
    },
  ];

  const list = [
    {
      id: "LLM",
      title: "Large Language Models",
    },
    {
      id: "DL",
      title: "Deep Learning",
    },
    {
      id: "CV",
      title: "Computer Vision",
    },
    {
      id: "other",
      title: "Other",
    },
  ];

  export const llmProject = [
    {
      name: "MediBot-GraphRAG",
      description:
        "GraphRag based chatbot built using LangChain and Neo4j, designed for hospital systems. The chatbot retrieves structured and unstructured data about patients, visits, physicians, insurance payers, and hospital locations.",
      tags: [
        {
          name: "Neo4j",
          color: "blue-text-gradient",
        },
        {
          name: "Langchain",
          color: "green-text-gradient",
        },
        {
          name: "Docker",
          color: "pink-text-gradient",
        },
      ],
      image: "https://realpython.com/cdn-cgi/image/width=1920,format=auto/https://files.realpython.com/media/Build-a-LLM-Application-with-LangChain_Watermarked.b9d023202ad2.jpg",
      source_link: "https://github.com/Jkanishkha0305/Medibot-GraphRAG",
      source_code_link: "https://github.com/Jkanishkha0305/Medibot-GraphRAG",
    },
    {
      name: "IKEA Assembly Assistant",
      description:
        "Developed a multimodal system that utilizes OpenAI GPT-4o, Gemini, and CLIP to analyze 10+ IKEA assembly manuals, extracting information from text and images. Used LlamaParse, and Cohere Reranker for precise retrieval of assembly instructions.",
      tags: [
        {
          name: "OpenAI API",
          color: "blue-text-gradient",
        },
        {
          name: "LLamaParse",
          color: "green-text-gradient",
        },
        {
          name: "Cohere API",
          color: "pink-text-gradient",
        },
      ],
      image: "https://realpython.com/cdn-cgi/image/width=1920,format=auto/https://files.realpython.com/media/ChatGPT-A-Great-Coding-Mentor-for-Learning-Python_Watermarked.3161825ae6b8.jpg",
      source_link: "https://github.com/Jkanishkha0305/End-to-End-LLM-Projects/tree/main/IKEA_Assembly_RAG",
      source_code_link: "https://github.com/Jkanishkha0305/End-to-End-LLM-Projects/tree/main/IKEA_Assembly_RAG",
    },
    {
      name: "Clinic AI Assistant",
      description:
        "Built a Private AI Assistant for clinics and hospitals which fetches patient data and answers questions on top of that data, using Qdrant Hybrid Cloud (JWT-RBAC), DSPy and Groq — Llama3",
      tags: [
        {
          name: "Qdrant Hybrid Cloud",
          color: "blue-text-gradient",
        },
        {
          name: "Groq — Llama3",
          color: "green-text-gradient",
        },
        {
          name: "DSPy",
          color: "pink-text-gradient",
        },
      ],
      image: "https://realpython.com/cdn-cgi/image/width=1920,format=auto/https://files.realpython.com/media/Linear-Regression-in-Python_Watermarked.479f82188ace.jpg",
      source_link: "https://github.com/Jkanishkha0305/End-to-End-LLM-Projects/tree/main/Groq_Clininc_Assitant",
      source_code_link: "https://github.com/Jkanishkha0305/End-to-End-LLM-Projects/tree/main/Groq_Clininc_Assitant",
    },
    
    {
      name: "LangQuery",
      description:
        "Built an end-to-end LLM project using Google PaLM and Langchain that enables natural language interaction with a MySQL database. Users can ask questions, and the system generates SQL queries to fetch data from the store’s inventory, sales, and discounts database.",
      tags: [
        {
          name: "GooglePaLM",
          color: "blue-text-gradient",
        },
        {
          name: "Langchain",
          color: "green-text-gradient",
        },
        {
          name: "ChromaDB",
          color: "pink-text-gradient",
        },
      ],
      image: "https://realpython.com/cdn-cgi/image/width=1920,format=auto/https://files.realpython.com/media/Embeddings-and-Vector-Databases-with-ChromaDB_Watermarked.646a2e85613a.jpg",
      source_link: "https://github.com/Jkanishkha0305/End-to-End-LLM-Projects/tree/main/PaLM_sqldb_tshirts",
      source_code_link: "https://github.com/Jkanishkha0305/End-to-End-LLM-Projects/tree/main/PaLM_sqldb_tshirts",
    },
    {
      name: "Multi Agent Financial Analysis",
      description:
        "Implemented a multi-agent system using CrewAI where agents like Financial News Analyst, Data Analyst, and Trading Strategy Agent collaborate on tasks. Each agent has defined roles, tools, and tasks with expected outputs.",
      tags: [
        {
          name: "SERPER API",
          color: "blue-text-gradient",
        },
        {
          name: "Langchain",
          color: "green-text-gradient",
        },
        {
          name: "CrewAI",
          color: "pink-text-gradient",
        },
      ],
      image: "https://realpython.com/cdn-cgi/image/width=1920,format=auto/https://files.realpython.com/media/How-to-Plot-With-Pandas_Watermarked.5bb0299e061b.jpg",
      source_link: "https://github.com/Jkanishkha0305/AI-Agents/blob/main/CrewAI_Multi_Agent_Collaboration.ipynb",
      source_code_link: "https://github.com/Jkanishkha0305/AI-Agents/blob/main/CrewAI_Multi_Agent_Collaboration.ipynb",
    },
    {
      name: "Ecommmerce Chatbot",
      description:
        "Fine-tuned Microsoft PHI3 using Unsloth Framework and Parameter-Efficient Fine-Tuning (PEFT) techniques to enhance its performance in E-Commerce chatbot applications, optimizing for improved contextual understanding and response generation.",
      tags: [
        {
          name: "Microsoft PHI3",
          color: "blue-text-gradient",
        },
        {
          name: "Unsloth",
          color: "green-text-gradient",
        },
        {
          name: "PEFT",
          color: "pink-text-gradient",
        },
      ],
      image: "https://realpython.com/cdn-cgi/image/width=1920,format=auto/https://files.realpython.com/media/How-to-Use-SpaCy-for-Natural-Language-Processing-in-Python_Watermarked_1.b363fc084a80.jpg",
      source_link: "https://github.com/Jkanishkha0305/End-to-End-LLM-Projects/blob/main/PHI3_Finetuning_Unsloth/README.md",
      source_code_link: "https://github.com/Jkanishkha0305/End-to-End-LLM-Projects/blob/main/PHI3_Finetuning_Unsloth/README.md",
    },
  ];
  
  export const cProject = [
    {
      name: "MediGenX",
      description:
        "Developed a CNN-based system for pneumonia detection from chest X-rays and a generative AI algorithm for personalized medical prescriptions. Fine-tuned a GPT-2 model to enhance prescription accuracy.",
      tags: [
        {
          name: "GAN",
          color: "blue-text-gradient",
        },
        {
          name: "Tensorflow",
          color: "green-text-gradient",
        },
        {
          name: "GPT-2",
          color: "pink-text-gradient",
        },
      ],
      image: "https://realpython.com/cdn-cgi/image/width=1920,format=auto/https://files.realpython.com/media/An-Introduction-to-Generative-Adversarial-Networks-GANs_Watermarked.6b71bfd66fda.jpg",
      source_link: "https://github.com/Jkanishkha0305/MediGenX",
      source_code_link: "https://github.com/Jkanishkha0305/MediGenX",
    },
    {
      name: "COVID-19 Detection",
      description:
        "Developed an end-to-end hybrid model combining CNN+LSTM layers to detect COVID-19 from chest X-rays. Conducted a comparative study with transfer learning models (Xception, MobileNet, VGG19). Built and deployed a web-app using Flask, hosted on Heroku.",
      tags: [
        {
          name: "PyTorch",
          color: "blue-text-gradient",
        },
        {
          name: "Flask",
          color: "green-text-gradient",
        },
        {
          name: "Heroku",
          color: "pink-text-gradient",
        },
      ],
      image: "https://realpython.com/cdn-cgi/image/width=1920,format=auto/https://files.realpython.com/media/K-Means-Clustering-in-Python_Watermarked.70101a29a2a2.jpg",
      source_link: "https://github.com/Jkanishkha0305/An-End-to-End-Hybrid-Learning-Model-for-Detection-of-Covid-19-from-CHest-Xray-Images",
      source_code_link: "https://github.com/Jkanishkha0305/An-End-to-End-Hybrid-Learning-Model-for-Detection-of-Covid-19-from-CHest-Xray-Images",
    },
    {
      name: "FinSentinAl",
      description:
        "Scraped financial articles, applied sentiment analysis, text summarization, analyzed stock correlations, and developed a forecasting model to correlate sentiments with stocks, using Plotly for visualization.",
      tags: [
        {
          name: "Beautiful Soup",
          color: "blue-text-gradient",
        },
        {
          name: "HuggingFace",
          color: "green-text-gradient",
        },
        {
          name: "yfinance API",
          color: "pink-text-gradient",
        },
      ],
      image: "https://realpython.com/cdn-cgi/image/width=1920,format=auto/https://files.realpython.com/media/How-to-use-NLTK-for-Sentiment-Analysis-in-Python_Watermarked.05ff07ca7ec7.jpg",
      source_link: "https://github.com/Jkanishkha0305/FinSentinAl",
      source_code_link: "https://github.com/Jkanishkha0305/FinSentinAl",
    },
    {
      name: "QuakeAI-Fusion",
      description:
        "Analyzed seismic parameters, applied time series techniques, developed forecasting models, used clustering and anomaly detection, and generated synthetic seismic data using VAE.",
      tags: [
        {
          name: "Time Series",
          color: "blue-text-gradient",
        },
        {
          name: "Anamoly Detection",
          color: "green-text-gradient",
        },
        {
          name: "VAE",
          color: "pink-text-gradient",
        },
      ],
      image: "https://realpython.com/cdn-cgi/image/width=1920,format=auto/https://files.realpython.com/media/Showcase-Folium_Watermarked.e23256df3c8d.jpg",
      source_link: "https://github.com/Jkanishkha0305/QuakeAi-Fusion",
      source_code_link: "https://github.com/Jkanishkha0305/QuakeAi-Fusion",
    },
    {
      name: "Malaria Detection",
      description:
        "Developed a malaria detection system using PyTorch and MobileNet for image classification. Built a web application with Flask and deployed it on Heroku for real-time detection.",
      tags: [
        {
          name: "PyTorch",
          color: "blue-text-gradient",
        },
        {
          name: "Flask",
          color: "green-text-gradient",
        },
        {
          name: "Heroku",
          color: "pink-text-gradient",
        },
      ],
      image: "https://realpython.com/cdn-cgi/image/width=1920,format=auto/https://files.realpython.com/media/UPDATE-Flask-by-Example-Project-Setup_Watermarked.f2d1b7a7a3d2.jpg",
      source_link: "https://github.com/Jkanishkha0305/Malaria-Detection-End-to-End",
      source_code_link: "https://github.com/Jkanishkha0305/Malaria-Detection-End-to-End",
    },
  ];
  
  export const webProject = [
    {
      name: "AutoDrive-Vision",
      description:
        "Developed deep learning and computer vision models for traffic signal classification, obstacle detection, and lane detection. Built an autonomous vehicle using Jetson Nano for lane detection, obstacle avoidance, and traffic signal response.",
      tags: [
        {
          name: "YOLO",
          color: "blue-text-gradient",
        },
        {
          name: "Mask-RCNN",
          color: "green-text-gradient",
        },
        {
          name: "Jetson Nano",
          color: "pink-text-gradient",
        },
      ],
      image: "https://realpython.com/cdn-cgi/image/width=1920,format=auto/https://files.realpython.com/media/A-Guide-to-Redis--Python_Watermarked.fadbf320f71f.jpg",
      source_link: "https://github.com/Jkanishkha0305/AutoDrive-Vision",
      source_code_link: "https://github.com/Jkanishkha0305/AutoDrive-Vision",
    },
    {
      name: "Automatic Attendance System",
      description:
        "Built an automatic attendance system using OpenCV, Python, and a face encoder with a cascade classifier for real-time face detection and recognition. The system records attendance, marking entry and exit times, and stores the data in CSV files.",
      tags: [
        {
          name: "OpenCV",
          color: "blue-text-gradient",
        },
        {
          name: "Python",
          color: "green-text-gradient",
        },
        {
          name: "Face Encoder",
          color: "pink-text-gradient",
        },
      ],
      image: "https://realpython.com/cdn-cgi/image/width=1920,format=auto/https://files.realpython.com/media/Face-Recognition-with-Python_Watermarked.b2d3b4911af3.jpg",
      source_link: "https://github.com/Jkanishkha0305/Attendance-system-using-OpenCv-and-Face-Recognition/tree/main",
      source_code_link: "https://github.com/Jkanishkha0305/Attendance-system-using-OpenCv-and-Face-Recognition/tree/main",
    },
    {
      name: "ASL Translater",
      description:
        "Built an ASL detection system using a custom CNN model, OpenCV and TensorFlow. Utilized the Kaggle ASL dataset and applied image masking techniques, and enhanced image quality with CLAHE normalization for improved detection accuracy.",
      tags: [
        {
          name: "OpenCV",
          color: "blue-text-gradient",
        },
        {
          name: "Tensorflow",
          color: "green-text-gradient",
        },
        {
          name: "Python",
          color: "pink-text-gradient",
        },
      ],
      image: "https://realpython.com/cdn-cgi/image/width=1920,format=auto/https://files.realpython.com/media/Bitwise-Operators-in-Python_Watermarked.85ff8fc6a931.jpg",
      source_link: "https://github.com/Jkanishkha0305/ASL-Detection-system-using-Custom-CNN-and-ML-Techniques",
      source_code_link: "https://github.com/Jkanishkha0305/ASL-Detection-system-using-Custom-CNN-and-ML-Techniques",
    },
    {
      name: "SkyBot",
      description:
        "Developed deep learning and computer vision model for mobile robot detection, compared Custom DCNN with YOLOv5 and YOLOv7, and deployed the best model on a DJI Tello drone. Used PID controller and OpenCV for tracking and following the robot.",
      tags: [
        {
          name: "dji Tello",
          color: "blue-text-gradient",
        },
        {
          name: "PID",
          color: "green-text-gradient",
        },
        {
          name: "PyTorch",
          color: "pink-text-gradient",
        },
      ],
      image: "https://realpython.com/cdn-cgi/image/width=1920,format=auto/https://files.realpython.com/media/Image-Processing-in-Python-With-Pillow_Watermarked.b86d7e55f981.jpg",
      source_link: "https://github.com/Jkanishkha0305/Visual-Tracking-of-Mobile-Robots-using-UAV",
      source_code_link: "https://github.com/Jkanishkha0305/Visual-Tracking-of-Mobile-Robots-using-UAV",
    },
  ];
  
  export const otherProject = [
    {
      name: "Student Performance Analysis",
      description:
        "Built an end-to-end ML model for student mark prediction with CI/CD pipelines using GitHub Actions and AWS CodeRunner. Deployed a Flask app on AWS Elastic Beanstalk, handling data ingestion, transformation, model training, and deployment.",
      tags: [
        {
          name: "CI/CD",
          color: "blue-text-gradient",
        },
        {
          name: "AWS Coderunner",
          color: "green-text-gradient",
        },
        {
          name: "CatBoost",
          color: "pink-text-gradient",
        },
      ],
      image: "https://realpython.com/cdn-cgi/image/width=1920,format=auto/https://files.realpython.com/media/What-is-Data-Engineering_Watermarked.607e761a3c0e.jpg",
      source_link: "https://github.com/Jkanishkha0305/ML-with-pipelining-for-deployment-SPA",
      source_code_link: "https://github.com/Jkanishkha0305/ML-with-pipelining-for-deployment-SPA",
    },
    {
      name: "Docker ML APP",
      description:
        "Developed an end-to-end bank note authenticity detection system using Docker and machine learning. The project includes data preprocessing, model training, and deployment within a Dockerized environment.",
      tags: [
        {
          name: "Docker",
          color: "blue-text-gradient",
        },
        {
          name: "Machine Learning",
          color: "green-text-gradient",
        },
        {
          name: "Flasgger_api",
          color: "pink-text-gradient",
        },
      ],
      image: "https://realpython.com/cdn-cgi/image/width=480,format=auto/https://files.realpython.com/media/Python-Docker-Tutorials_Watermarked.f9834dc9df9a.jpg",
      source_link: "https://github.com/shinchancode/Mini-Project-SQL-LibraryManagement",
      source_code_link: "https://github.com/shinchancode/Mini-Project-SQL-LibraryManagement",
    },
    {
      name: "Coding Problems",
      description:
        "Solutions of various coding problems from platforms like DeepML, LeetCode, Codeforces, HackerRank, and GeeksforGeeks. The repository includes solutions in Python, SQL and C++.",
      tags: [
        {
          name: "SQL",
          color: "blue-text-gradient",
        },
        {
          name: "Python",
          color: "green-text-gradient",
        },
        {
          name: "C++",
          color: "pink-text-gradient",
        },
      ],
      image: "https://realpython.com/cdn-cgi/image/width=1920,format=auto/https://files.realpython.com/media/VSCode--Python-for-Advanced-Users_Watermarked.8fc072163645.jpg",
      source_link: "https://github.com/Jkanishkha0305/Coding-Problems",
      source_code_link: "https://github.com/Jkanishkha0305/Coding-Problems",
    },
    {
      name: "Bioinformatics - Drug Discovery",
      description:
        "Used ChEMBL web service to retrieve bioactivity data for SARS coronavirus 3C proteinase. Performed EDA, calculated Lipinski and PaDEL, and trained models using random forest and other ML algorithms for bioactivity prediction.",
      tags: [
        {
          name: "machine learning",
          color: "blue-text-gradient",
        },
        {
          name: "ChEMBL",
          color: "green-text-gradient",
        },
        {
          name: "Streamlit",
          color: "pink-text-gradient",
        },
      ],
      image: "https://realpython.com/cdn-cgi/image/width=1920,format=auto/https://files.realpython.com/media/Linear-Regression-in-Python_Watermarked.479f82188ace.jpg",
      source_link: "https://github.com/Jkanishkha0305/Bio-informatics-project",
      source_code_link: "https://github.com/Jkanishkha0305/Bio-informatics-project",
    },
    {
      name: "Transformers from Scratch",
      description:
        "A Github Repo developed to build different Transformer architecture like - (GPT, Diffusion, CLIP, CLAP, BLIP) from scratch using PyTorch, Python, and HuggingFace. The project includes building the encoder, decoder, attention mechanism, and positional encoding.",
      tags: [
        {
          name: "PyTorch",
          color: "blue-text-gradient",
        },
        {
          name: "Python",
          color: "green-text-gradient",
        },
        {
          name: "HuggingFace",
          color: "pink-text-gradient",
        },
      ],
      image: "https://realpython.com/cdn-cgi/image/width=1920,format=auto/https://files.realpython.com/media/Pytorch-vs-Tensorflow_Watermarked.9975d1999917.jpg",
      source_link: "https://github.com/Jkanishkha0305/Transformers-from-Scratch",
      source_code_link: "https://github.com/Jkanishkha0305/Transformers-from-Scratch",
    },
    {
      name: "Gallery AI",
      description:
        "Created an Intelligent Image Gallery with Upload feature, Deduplication, and Text-Based image Search Using Vector DB Qdrant, OpenAI CLIP embeddings and Sentence Transformers.",
      tags: [
        {
          name: "clip-ViT-B-32",
          color: "blue-text-gradient",
        },
        {
          name: "Qdrant",
          color: "green-text-gradient",
        },
        {
          name: "HuggingFace",
          color: "pink-text-gradient",
        },
      ],
      image: "https://realpython.com/cdn-cgi/image/width=1920,format=auto/https://files.realpython.com/media/Generate-Images-with-DALL-E-2-and-OpenAIs-API_Watermarked.05753350e866.jpg",
      source_link: "https://github.com/Jkanishkha0305/End-to-End-LLM-Projects/tree/main/CLIP_AI_Gallery",
      source_code_link: "https://github.com/Jkanishkha0305/End-to-End-LLM-Projects/tree/main/CLIP_AI_Gallery",
    },
    {
      name: "MarketAI",
      description:
        "Built a user-friendly news research tool designed for effortless information retrieval. Users can input article URLs and ask questions to receive relevant insights from the stock market and financial domain.",
      tags: [
        {
          name: "Open AI API",
          color: "blue-text-gradient",
        },
        {
          name: "Langchain",
          color: "green-text-gradient",
        },
        {
          name: "FAISS",
          color: "pink-text-gradient",
        },
      ],
      image: "https://realpython.com/cdn-cgi/image/width=1920,format=auto/https://files.realpython.com/media/Monthly-Python-News_Purple_Watermarked.5b2e306328cb.jpg",
      source_link: "https://github.com/Jkanishkha0305/End-to-End-LLM-Projects/tree/main/PaLM_news_research_tool",
      source_code_link: "https://github.com/Jkanishkha0305/End-to-End-LLM-Projects/tree/main/PaLM_news_research_tool",
    },
    
  ];
  
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
      icon: "https://ui-avatars.com/api/?name=KPTAC&background=f3343c&color=ffffff&bold=true&size=256",
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
