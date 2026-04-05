import React, { useRef } from "react";
import { motion } from "framer-motion";

import { styles } from "../styles";
import { SectionWrapper } from "../hoc";
import { fadeIn, textVariant, staggerContainer } from "../utils/motion";
import { research } from "../constants";
import "./Research.scss";

const Research = () => {
  const scrollerRef = useRef(null);

  const scrollCards = (direction) => {
    const el = scrollerRef.current;
    if (!el) return;
    const card = el.querySelector(".research-card");
    const cardWidth = card ? card.getBoundingClientRect().width + 20 : el.clientWidth * 0.9;
    const step = el.clientWidth >= 1024 ? cardWidth * 3 : el.clientWidth >= 700 ? cardWidth * 2 : cardWidth;
    el.scrollBy({ left: direction * step, behavior: "smooth" });
  };

  const getCrispSummary = (text, maxLength = 180) => {
    if (!text) return "";
    const normalized = text.replace(/\s+/g, " ").trim();
    const sentences = normalized.match(/[^.!?]+[.!?]?/g) || [normalized];
    const candidate = sentences.slice(0, 2).join(" ").trim();
    if (candidate.length <= maxLength) return candidate;
    return `${normalized.slice(0, maxLength - 3).trimEnd()}...`;
  };

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
        <div className='carousel-toolbar'>
          <button
            type='button'
            className='carousel-btn'
            onClick={() => scrollCards(-1)}
            aria-label='Scroll research left'
          >
            ←
          </button>
          <button
            type='button'
            className='carousel-btn'
            onClick={() => scrollCards(1)}
            aria-label='Scroll research right'
          >
            →
          </button>
        </div>
        <motion.ul
          ref={scrollerRef}
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
              whileHover={{ y: -4 }}
            >
              {(() => {
                const tags = item.tags || [];
                const advisorTags = tags.filter((tag) => /prof advisor/i.test(tag));
                const otherTags = tags.filter((tag) => !/prof advisor/i.test(tag));
                const visibleOtherTags = otherTags.slice(0, 2);
                const hiddenCount = otherTags.length - visibleOtherTags.length;
                const visibleTags = [...visibleOtherTags, ...advisorTags];

                return (
                  <>
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
                <p className='research-summary'>{getCrispSummary(item.summary, 140)}</p>
              )}

              {visibleTags.length > 0 && (
                <div className='research-tags'>
                  {visibleTags.map((tag) => (
                    <span key={`${item.title}-${tag}`} className='research-tag'>
                      {tag}
                    </span>
                  ))}
                  {hiddenCount > 0 && (
                    <span className='research-tag muted'>+{hiddenCount} more</span>
                  )}
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
                  </>
                );
              })()}
            </motion.li>
          ))}
        </motion.ul>
      </div>
    </div>
  );
};

export default SectionWrapper(Research, "");
