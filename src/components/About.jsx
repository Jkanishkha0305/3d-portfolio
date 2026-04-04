import React from "react";
import { motion } from "framer-motion";

import { styles } from "../styles";
import { SectionWrapper } from "../hoc";
import { fadeIn, textVariant } from "../utils/motion";
import "./About.scss";

const About = () => {
    const focusAreas = [
      "Multi-Agent Workflows",
      "LLM Applications",
      "Healthcare AI",
      "Enterprise ML",
      "Production Reliability",
    ];

    return (
      <>
        <motion.div variants={textVariant()}>
          <p className={styles.sectionSubText}>Introduction</p>
          <h2 className={styles.sectionHeadText}>Overview.</h2>
        </motion.div>
  
        <motion.div
          variants={fadeIn("", "", 0.1, 1)}
          className="about-panel mt-6 text-secondary text-[17px] w-full"
        >
          <div className='about-panel__header'>
            <span className='about-kicker'>Machine Learning Engineer</span>
            <span className='about-badge'>Research → Production</span>
          </div>

          <p className='about-lead'>
            I build AI systems that move from research to production.
          </p>

          <p className='about-copy'>
            Across NYU Center for Data Science, Toyota, Genmab, and startups,
            I have built multi-agent workflows, LLM applications, and practical ML
            products for healthcare and enterprise.
          </p>

          <p className='about-copy'>
            I care about clear problem framing, measurable outcomes, and engineering
            that stays reliable outside the notebook.
          </p>

          <div className='about-focus'>
            {focusAreas.map((item) => (
              <span key={item} className='about-focus-chip'>
                {item}
              </span>
            ))}
          </div>
        </motion.div>
        
      </>
    );
  };

export default SectionWrapper(About, "");
