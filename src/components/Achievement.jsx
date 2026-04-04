import React, { useRef } from "react";
import { motion } from "framer-motion";

import { styles } from "../styles";
import { SectionWrapper } from "../hoc";
import { fadeIn, textVariant } from "../utils/motion";
import { achievements} from "../constants"; // Import the research list
import "./Achievement.scss";

const Achievement = () => {
  const scrollerRef = useRef(null);

  const scrollCards = (direction) => {
    const el = scrollerRef.current;
    if (!el) return;
    const card = el.querySelector(".achievement-card");
    const cardWidth = card ? card.getBoundingClientRect().width + 20 : el.clientWidth * 0.9;
    const step = el.clientWidth >= 1024 ? cardWidth * 4 : el.clientWidth >= 700 ? cardWidth * 2 : cardWidth;
    el.scrollBy({ left: direction * step, behavior: "smooth" });
  };

  return (
    <div className={`mt-12 bg-black-100 rounded-[20px]`}>
      {/* Achievements Section */}
      <div className={`bg-tertiary rounded-2xl ${styles.padding}`}>
        <motion.div variants={textVariant()}>
          <p className={styles.sectionSubText}>Some Glimpses on...</p>
          <h2 className={styles.sectionHeadText}>Achievements.</h2>
        </motion.div>
      </div>
      <div className={`-mt-20 justify-center p-6 ${styles.paddingX}`}>
        <div className='carousel-toolbar'>
          <button
            type='button'
            className='carousel-btn'
            onClick={() => scrollCards(-1)}
            aria-label='Scroll achievements left'
          >
            ←
          </button>
          <button
            type='button'
            className='carousel-btn'
            onClick={() => scrollCards(1)}
            aria-label='Scroll achievements right'
          >
            →
          </button>
        </div>
        <motion.ul
          ref={scrollerRef}
          className='achievement-grid'
          variants={fadeIn("", "spring", 0.1, 0.8)}
        >
          {achievements.map((achievement, index) => (
            <motion.li
              key={achievement.title}
              className='achievement-card'
              variants={fadeIn("up", "spring", index * 0.1, 0.6)}
              whileHover={{ y: -4 }}
            >
              <span className='achievement-pill'>
                {String(index + 1).padStart(2, "0")}
              </span>
              <p className='achievement-text'>{achievement.title}</p>
            </motion.li>
          ))}
        </motion.ul>
      </div>
    </div>
  );
};

export default SectionWrapper(Achievement, "");
