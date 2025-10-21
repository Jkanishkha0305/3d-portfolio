import React from 'react';
import './FloatingParticles.scss';

const FloatingParticles = () => {
  const particles = Array.from({ length: 20 }, (_, i) => i);

  return (
    <div className="floating-particles-container">
      {particles.map((i) => (
        <div key={i} className="particle"></div>
      ))}
    </div>
  );
};

export default FloatingParticles;
