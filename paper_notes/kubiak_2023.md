# TTV Constraints on Additional Planets in the WD 1856+534 system
- Paper link: https://arxiv.org/abs/2303.06157
- Common envelope evolution: planet survives star's red giant phase by ejecting envelope so that it stops falling inward
    - Challenges: low mass and long period compared to other planets known to have undergone common envelope evolution
- High eccentricity migration: gravitational interactions with other bodies in the system excited planet onto highly eccentric orbit, which is then circularized at a small orbital distance due to tidal interactions
- ttvfast software (https://github.com/kdeck/TTVFast) used to compute expected WD 1856+534 b transit times under various assumptions about a (hypothetical) companion
    - ttvfast assumes planets are massless, which is a decent but imperfect approximation for this system; account for this by adding planet's assumed mass to star's mass
    - ttvfast ignores Romer delay, which can be computed separately and added to the ttvfast predictions
- Orbital period measured from observations over a short baseline could be different from orbital period measured from observations over a short time (see paper for method to work around this)
- Due to computational constraints, perform grid search based on hypothetical companion's mass, period, eccentricity, and inclination (with mean anomaly
at start of integration and argument of periastron treated as nuisance parameters)
- Best-fit two-planet model has higher likelihood than single planet model, but is not the best explanation for the observed transit times based on BIC (which accounts for the fact that two-planet model also has more parameters)
- Hypothetical companion sufficiently massive to perturb WD 1856+534 b onto present orbit would *not* be detectable with the baseline (just over a year) of transits included in this study
- Including data from small telescopes in addition to Gemini and GTC improves constraints very slightly