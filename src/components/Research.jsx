import React from "react";
import { motion } from "framer-motion";

import { styles } from "../styles";
import { SectionWrapper } from "../hoc";
import { fadeIn, textVariant } from "../utils/motion";
import { research } from "../constants"; // Import the research list
import "./Achievement.scss";

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
      <div className={`-mt-20 justify-center p-6 ${styles.paddingX} gap-7`}>
        <ul className='mt-5 list-disc ml-5 space-y-2'>
          {research.map((research, index) => (
            <li key={index} className='text-white-100 text-[15px] pl-1'>
              {/* Embed the link in the title */}
               
              {research.link ? (
                <>
                  {research.title}
                  <a
                    href={research.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-400 underline"
                  >
                    Link
                  </a>
                </>
              ) : (
                <>
                  {research.title}
                </>
              )}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default SectionWrapper(Research, "");