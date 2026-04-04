import { BrowserRouter } from "react-router-dom";

import {Contact, Profile, About, Achievement, Experience, Education, Hero, Navbar, Tech, Project, Research, StarsCanvas, Content, Footer } from "./components";

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
        </div>
        <div className='relative z-0'>
          <Education />
        </div>
        <div className='relative z-0'>
          <Project />
        </div>
        <div className='relative z-0'>
          <Experience />
        </div>
        <div className='relative z-0'>
          <Achievement />
        </div>
        <div className='relative z-0'>
          <Research />
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
