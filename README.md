<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Yalton/TradingBot">
    <img src="doc/img/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Tradebert</h3>

  <p align="center">
    project_description
    <br />
    <a href="https://github.com/Yalton/TradingBot"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/Yalton/TradingBot">View Demo</a>
    ·
    <a href="https://github.com/Yalton/TradingBot/issues">Report Bug</a>
    ·
    <a href="https://github.com/Yalton/TradingBot/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

Introducing the Tradebert, a powerful and sophisticated neural network trading bot meticulously crafted in Python. This cutting-edge AI-driven solution harnesses the potential of web scraping to gather crucial financial data from the internet, which enables it to make informed and intelligent trading decisions.

Designed with a deep learning architecture, Tradebert is capable of analyzing vast amounts of data from multiple sources, including financial news, market trends, and historical prices. By using advanced natural language processing techniques, the bot comprehends textual data and extracts meaningful insights to fuel its decision-making process.

Tradebert's core features include:

1. Dynamic Web Scraping: The bot continually scrapes relevant financial information from reputable sources, ensuring that it stays updated with the latest market developments.
2. Intelligent Data Processing: Tradebert cleanses and preprocesses the collected data, converting it into a structured format that's easily digestible by the neural network.
3. Advanced Natural Language Processing: Equipped with state-of-the-art NLP algorithms, the bot comprehends and analyzes textual information to derive critical insights that influence its trading strategies.
4. Market Sentiment Analysis: By gauging public sentiment and investor confidence, Tradebert is able to make well-informed predictions about the future performance of specific assets.
5. Customizable Trading Strategies: Tradebert is adaptable to various trading styles and risk appetites, enabling users to tailor their strategies to align with their financial goals and preferences.
6. Real-time Decision Making: With its robust neural network architecture, Tradebert rapidly processes information and makes split-second trading decisions to capitalize on emerging market opportunities.
7. Comprehensive Performance Metrics: Users have access to a detailed dashboard that provides real-time insights into the bot's performance, including key metrics such as return on investment (ROI), win/loss ratio, and overall profitability.
8. Intuitive User Interface: Tradebert is designed with usability in mind, featuring a sleek, user-friendly interface that allows even novice traders to navigate the platform with ease.

Embrace the power of AI-driven trading with Tradebert, the ultimate Python-based neural network trading bot that seamlessly combines web scraping and advanced analytics to revolutionize your investment experience.

### Available Technical Indicators 

- relative_strength_index
- moving_average
- exponential_moving_average
- rate_of_change
- stochastic_oscillator
- fibonnaci_levels
- on_balance_volume
- ichimoku_cloud
- average_true_range
- Directional Movement Index (DMI)
- Chaikin Money Flow (CMF):
- Gann Angles
- Volume-Weighted Average Price (VWAP):
- Pivot Points

### Available exchanges 

- Binance 
- Kucoin
- Alpaca 

Alpaca is mostly useful for their paper sockets, as the exchange only offers spot trading the potential returns are only 1x

When testing trading strategies (or implementing your own) it is highly recommended to use the Paper sockets at first in order to validate and backtest 

### Available trading strategies  

Moving Average Convergence Divergence Crossover Strategy 

Spueeze Momentum strategy 

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![Next][Next.js]][Next-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
<!-- ## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps. -->

<!-- ### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/Yalton/TradingBot.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ``` -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
 ## Roadmap

- [ ] Webscraping social media and performing sentiment analysis to analyze assets 
- [ ] Predict using brownian motion
- [ ] Give score of asset based on each quant fi analysis
- [ ] Use score to predict whether to sell, buy, long, short an asset
- [ ] Improve asset analysis models
- [ ] Calculate line of support
- [ ] Calculate line of resistance
- [ ] Use technical indicators to determine if market is trending or sideways
- [ ] Use confluence of technical indicators to determine signals
- [ ] Fix HMM
- [ ] Integrate GPT4All
- [ ] Fix Brownian motion
- [ ] Add an item

See the [open issues](https://github.com/Yalton/TradingBot/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

 -->

<!-- CONTRIBUTING -->
<!-- ## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Dalton Bailey- [@yalt7117](https://twitter.com/@yalt7117) - drbailey117@gmail.com

Project Link: [https://github.com/Yalton/TradingBot](https://github.com/Yalton/TradingBot)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
<!-- ## Acknowledgments

* []()
* []()
* []() -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/Yalton/TradingBot.svg?style=for-the-badge
[contributors-url]: https://github.com/Yalton/TradingBot/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Yalton/TradingBot.svg?style=for-the-badge
[forks-url]: https://github.com/Yalton/TradingBot/network/members
[stars-shield]: https://img.shields.io/github/stars/Yalton/TradingBot.svg?style=for-the-badge
[stars-url]: https://github.com/Yalton/TradingBot/stargazers
[issues-shield]: https://img.shields.io/github/issues/Yalton/TradingBot.svg?style=for-the-badge
[issues-url]: https://github.com/Yalton/TradingBot/issues
[license-shield]: https://img.shields.io/github/license/Yalton/TradingBot.svg?style=for-the-badge
[license-url]: https://github.com/Yalton/TradingBot/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/dalton-r-bailey
[product-screenshot]: doc/img/Tradebert.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 