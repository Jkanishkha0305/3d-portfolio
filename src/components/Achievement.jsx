import React from "react";
import { motion } from "framer-motion";

import { styles } from "../styles";
import { SectionWrapper } from "../hoc";
import { fadeIn, textVariant } from "../utils/motion";
import { achievements} from "../constants"; // Import the research list
import "./Achievement.scss";

const Achievement = () => {
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
        <motion.ul
          className='achievement-grid'
          variants={fadeIn("", "spring", 0.1, 0.8)}
        >
          {achievements.map((achievement, index) => (
            <motion.li
              key={achievement.title}
              className='achievement-card'
              variants={fadeIn("up", "spring", index * 0.1, 0.6)}
              whileHover={{ y: -6, scale: 1.01 }}
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
