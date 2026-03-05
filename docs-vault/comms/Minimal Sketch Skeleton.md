Purpose: Scaffold for multi-turn AI completions
Assumptions: The multi-turn task being undertaken by the multi-turn agent can be constructed as an elimination process where eliminations are never revisited. 

On a page providing more details:

Modifications can accommodate revised eliminations - but CMBS is stateless about this. CMBS would view this as a new task while the systems using CMBS would still draw this as one overarching task. 

Example: 

You have an SRE agent that responds to outages.

**Symptom:**  
Users report 500 errors. PagerDuty fires:

> “Database CPU > 95% for 5 minutes.”

You build a hypothesis lattice like this:

H1: DB CPU saturation  
H2: DB connection pool exhaustion  
H3: DB disk full  
H4: App deploy regression  
H5: Network partition

You now start eliminating using CMBS (adding constraints from evidence).

You ssh and instead get CPU is 12%.

# The Revived Hypothesis

You must reintroduce:

H6: Monitoring data is wrong

And possibly revive:

H5: Network partition (metrics scraped from wrong node)  
H2: Connection pool exhaustion

You reopen the eliminated hypotheses because your elimination was predicated on faulty evidence.

