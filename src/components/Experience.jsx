import React from "react";
import {
  VerticalTimeline,
  VerticalTimelineElement,
} from "react-vertical-timeline-component";
import { motion } from "framer-motion";

import "react-vertical-timeline-component/style.min.css";
import "./Experience.scss";

import { styles } from "../styles";
import { experiences } from "../constants";
import { SectionWrapper } from "../hoc";
import { textVariant } from "../utils/motion";

const ExperienceCard = ({ experience }) => {
  return (
    <VerticalTimelineElement
      contentStyle={{
        background: "#1d1836",
        color: "#fff",
      }}
      contentArrowStyle={{ borderRight: "7px solid  #232631" }}
      iconStyle={{ background: experience.iconBg }}
      icon={
        <div className='experience-icon-shell'>
          <img
            src={experience.icon}
            alt={experience.company_name}
            className='experience-icon-mark'
          />
        </div>
      }
    >
      <div className='experience-header'>
        <div className='experience-copy'>
        <h3 className='experience-title text-white text-[24px] font-bold'>{experience.title}</h3>
        <p
          className='text-secondary text-[16px] font-semibold'
          style={{ margin: 0 }}
        >
          {experience.company_name}
        </p>
        </div>
        <div className='experience-date-chip'>{experience.date}</div>
      </div>

      <ul className='mt-5 ml-0 space-y-2'>
        {experience.points.map((point, index) => (
          <li
            key={`experience-point-${index}`}
            className='experience-point text-white-100 text-[14px] tracking-wider'
          >
            {point}
          </li>
        ))}
        {experience.link && (
          <li className='flex justify-start'>
            <a
              href={experience.link}
              className='experience-link blue-text-gradient'
              target="_blank"
              rel="noopener noreferrer"
            >
              Visit
            </a>
          </li>
        )}
      </ul>
    </VerticalTimelineElement>
  );
};

const Experience = () => {
  return (
    <>
      <motion.div
        whileInView={{ opacity: 1, transform: "none" }}
        id='experience'
        variants={textVariant()}
        style={{ scrollMarginTop: "150px" }}
      >
        <p className={`${styles.sectionSubText}`}>
          What I have done so far
        </p>
        <h2 className={`${styles.sectionHeadText}`}>
          Work Experience.
        </h2>
      </motion.div>

      <div className='mt-20 flex flex-col'>
        <VerticalTimeline lineColor='rgba(145, 94, 255, 0.28)'>
          {experiences.map((experience, index) => (
            <ExperienceCard
              key={`experience-${index}`}
              experience={experience}
            />
          ))}
        </VerticalTimeline>
      </div>
    </>
  );
};

export default SectionWrapper(Experience, "");
