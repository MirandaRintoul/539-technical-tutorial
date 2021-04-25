<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS_HTML">
  MathJax.Hub.Config({
    "HTML-CSS": {
      availableFonts: ["TeX"],
    },
    tex2jax: {
      inlineMath: [['$','$'],["\\(","\\)"]]},
      displayMath: [ ['$$','$$'], ['\[','\]'] ],
    TeX: {
      extensions: ["AMSmath.js", "AMSsymbols.js", "color.js"],
      equationNumbers: {
        autoNumber: "AMS"
      }
    },
    showProcessingMessages: false,
    messageStyle: "none",
    imageFont: null,
    "AssistiveMML": { disabled: true }
  });
</script>

# Expectation-Maximization (EM) Algorithm for NLP
Miranda Rintoul

Ling 539

## Background

In statistics and its applications, maximum likelihood estimation (MLE) is a popular technique for estimating the parameters of a distribution from observed data. The likelihood function on a parameter <img src="https://render.githubusercontent.com/render/math?math=\theta"> given data <img src="https://render.githubusercontent.com/render/math?math=y"> is equivalent to the probability density function of <img src="https://render.githubusercontent.com/render/math?math=y"> with parameter <img src="https://render.githubusercontent.com/render/math?math=\theta">.

<img src="https://render.githubusercontent.com/render/math?math=L(\theta|y) = f(y|\theta)">

The maximum likelihood estimator of <img src="https://render.githubusercontent.com/render/math?math=\theta"> is the argmax of the likelihood function.  Intuitively, it can be thought of as the parameter that is most likely to have generated the data.  In practice, it is conveneient to instead maximize the log of the likelihood function, <img src="https://render.githubusercontent.com/render/math?math=\ell(\theta|y)">.
