import { BrowserRouter } from "react-router-dom";

import {Contact, Profile, About, Achievement, Experience, Education, Hero, Navbar, Tech, Project, Research, StarsCanvas, Content, Footer, FloatingParticles } from "./components";

const App = () => {

  return (
    <div>
    <BrowserRouter>
      <div className='relative z-0 bg-primary'>
        <div className='bg-hero-pattern bg-cover bg-no-repeat bg-center'>
          <Navbar />
          <Hero />
        </div>
        <Content />
        <div className='relative z-0'>
          <About />
          <FloatingParticles />
        </div>
        <div className='relative z-0'>
          <Education />
          <FloatingParticles />
        </div>
        <div className='relative z-0'>
          <Project />
          <FloatingParticles />
        </div>
        <div className='relative z-0'>
          <Experience />
          <FloatingParticles />
        </div>
        <div className='relative z-0'>
          <Achievement />
          <FloatingParticles />
        </div>
        <div className='relative z-0'>
          <Research />
          <FloatingParticles />
        </div>
        <Profile/>
        <Tech />
        <div className='relative z-0'>
          <Contact />
          <StarsCanvas />
        </div>
        <Footer/>
      </div>
    </BrowserRouter>
    </div>
  )
}

export default App
