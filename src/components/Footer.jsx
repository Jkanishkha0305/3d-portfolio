import React from "react";
import { useEffect, useState } from "react";
import "./Footer.scss";

const Footer = () => {

    return (
        <div className="footer text-center">
        <p> 
          &#x3c;&#47;&#x3e; with ❤️ by
          <a href="https://jkanishkha0305.github.io" target="_blank" rel="noopener noreferrer">
            {" "}
            Kanishkha Jaisankar
          </a>
          😎
        </p>
        <p className="pink-text-gradient">No. of Visitors | <img className="visitcounter" src="https://hitwebcounter.com/counter/counter.php?page=9795911&style=0025&nbdigits=5&type=page&initCount=459" title="Counter Widget" alt="Visit counter For Websites"   border="0" /></p>

      </div>
    );
  };
  
  export default Footer;
