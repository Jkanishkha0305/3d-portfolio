import React from "react";
import { motion } from "framer-motion";

import { styles } from "../styles";
import { SectionWrapper } from "../hoc";
import { fadeIn, textVariant } from "../utils/motion";
import { about} from "../constants"; 
// import "./Achievement.scss";

const About = () => {
    return (
      <>
        <motion.div variants={textVariant()}>
          <p className={styles.sectionSubText}>Introduction</p>
          <h2 className={styles.sectionHeadText}>Overview.</h2>
        </motion.div>
  
        <motion.p
          variants={fadeIn("", "", 0.1, 1)}
          className="mt-4 text-secondary text-[17px] max-w-5xl leading-[30px]" style={{ textAlign: "justify" }}
        >
          {about.description1} <br />
          {about.description2} <br />
          {about.description3} <br />
          {about.description4}
        </motion.p>
        
      </>
    );
  };

export default SectionWrapper(About, "");