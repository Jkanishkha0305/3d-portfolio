import React from "react";
import { motion } from "framer-motion";

import { styles } from "../styles";
import { SectionWrapper } from "../hoc";
import { fadeIn, textVariant, staggerContainer } from "../utils/motion";
import { research } from "../constants";
import "./Research.scss";

const Research = () => {
  return (
    <div className={`mt-12 bg-black-100 rounded-[20px]`}>
      
      {/* Research Section */}
      <div className={`bg-tertiary rounded-2xl mt-10 ${styles.padding}`}>
        <motion.div variants={textVariant()}>
          <p className={styles.sectionSubText}>In the Field of...</p>
          <h2 className={styles.sectionHeadText}>Research.</h2>
        </motion.div>
      </div>
      <div className={`-mt-20 justify-center p-6 ${styles.paddingX}`}>
        <motion.ul
          className='research-grid'
          variants={staggerContainer(0.12, 0.2)}
          initial='hidden'
          whileInView='show'
          viewport={{ once: true, amount: 0.25 }}
        >
          {research.map((item, index) => (
            <motion.li
              key={item.title}
              className='research-card'
              variants={fadeIn("up", "spring", 0, 0.6)}
              whileHover={{ y: -6, scale: 1.01 }}
            >
              <div className='research-card__header'>
                {item.status && (
                  <span className='research-status'>{item.status}</span>
                )}
                {item.year && <span className='research-year'>{item.year}</span>}
              </div>

              <h3 className='research-title'>{item.title}</h3>

              {item.venue && (
                <p className='research-venue'>{item.venue}</p>
              )}

              {item.summary && (
                <p className='research-summary'>{item.summary}</p>
              )}

              {item.tags && item.tags.length > 0 && (
                <div className='research-tags'>
                  {item.tags.map((tag) => (
                    <span key={`${item.title}-${tag}`} className='research-tag'>
                      {tag}
                    </span>
                  ))}
                </div>
              )}

              {(item.link || item.repository) && (
                <div className='research-actions'>
                  {item.link && (
                    <a
                      href={item.link}
                      target="_blank"
                      rel="noopener noreferrer"
                      className='research-link'
                    >
                      {item.linkLabel || "View publication"}
                    </a>
                  )}
                  {item.repository && (
                    <a
                      href={item.repository}
                      target="_blank"
                      rel="noopener noreferrer"
                      className='research-link secondary'
                    >
                      {item.repositoryLabel || "View project"}
                    </a>
                  )}
                </div>
              )}
            </motion.li>
          ))}
        </motion.ul>
      </div>
    </div>
  );
};

export default SectionWrapper(Research, "");
