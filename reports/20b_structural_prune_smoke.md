# 20B structural prune smoke

## Variants

- general_50pct_experts: `/root/model/artifacts/20b_pruned_models/general_50pct_experts`
- math_25pct_experts: `/root/model/artifacts/20b_pruned_models/math_25pct_experts`

## Load + config checks

- general: dtype=torch.bfloat16 experts=16 top_k=4 ok=True
- math: dtype=torch.bfloat16 experts=8 top_k=4 ok=True

## Router stats (summary)

- general router tokens/s (forward-only, 5 prompts): 225
- math router tokens/s (forward-only, 5 prompts): 290

Top experts (layer 0):

- general: [0, 9, 10, 2, 4, 3, 11, 14]
- math: [2, 4, 6, 1, 3, 7, 5, 0]

Mean selected prob (layer 0):

- general: 0.2500
- math: 0.2500

## Generation (first 5 prompts)

### general_50pct_experts

**Prompt:** Solve: 17*23. Show work.

```text
**Problem**  
Evaluate the product  

\[
17 \times 23
\]

---

### 1.  Identify the operation  
We are multiplying two integers: 17 and 23.

### 2.  Perform the multiplication  
\[
17 \times 23 = 17 \cdot 23
\]

### 3.  Compute the result  
\[
17 \times 23 = 17 \times 23 = 17 \times 
```

**Prompt:** Explain what Mixture-of-Experts routing is in one paragraph.

```text
Mixture‑of‑Experts routing is a strategy for routing data in a network that uses several “expert” routing algorithms, each of which is best for a particular type of traffic or network condition.  When a packet arrives, the routing decision is made by selecting the most suitable expert algorithm for that packet, rather than using a single routing algorithm for all traffic.  This approach can improve routing performance by matching the routing algorithm to the packet’s characteristics (e.g., size,
```

**Prompt:** Write a Python function to compute Fibonacci numbers iteratively.

```text
Below is a simple Python function that computes the Fibonacci sequence iteratively.  
It starts with the first two numbers (0 and 1) and then repeatedly adds the
previous two numbers to get the next one.  
The function can be used to generate the first *n* Fibonacci numbers or to
continue the sequence indefinitely.

```python
def fibonacci_iterative(n):
    """
    Compute the first n Fibonacci numbers iteratively.

    Parameters:
        n  (int)
```

**Prompt:** Use a tool to get weather in SF. (Just describe the tool call you would make.)

```text
Sure! If you want to get the current weather for San Francisco, you could call the **Weather API** in the `weather` module. The call would look like this:

```text
weather.get("San Francisco")
```

This will return the latest temperature, precipitation, wind speed, and other relevant weather data for San Francisco.<|return|>
```

**Prompt:** Prove that the sum of two even numbers is even.

```text
**Proof that the sum of two even numbers is even**

Let \(a\) and \(b\) be two even integers.  
By definition, an even integer is a number that is divisible by \(2\); that is,

\[
a \equ 0 \pmod 2,\qqun b \equ 0 \pmod 2 .
\]

We consider the sum

\[
s = a + b .
\]

We want to show that
```


### math_25pct_experts

**Prompt:** Solve: 17*23. Show work.

```text
**Answer**: 17 × 23 = 391.<|return|>
```

**Prompt:** Explain what Mixture-of-Experts routing is in one paragraph.

```text
Mixture‑of‑Experts routing (MoE‑routing) is a routing scheme that uses a “mix‑of‑exper” (i.e., a mixture of experts) to route a network of a different routing algorithm that uses a “mix‑of‑exper” (i.e., a mixture of experts) to route a network of a different routing algorithm that uses a “mix‑of‑exper” (i.e., a mixture of experts) to route a network
```

**Prompt:** Write a Python function to compute Fibonacci numbers iteratively.

```text
Sure!  Below is a Python‑style function that iterates the Fibonacci sequence.  It simply starts with two values (0 and 1) and then adds the previous value to the next value, producing the next Fibonacci number.  The function can run a lot of iterations, so it’s easy to read and understand.  Feel free to tweak or add extra logic if you need to compute a large number or a specific Fibonacci sequence.

```python
def fibonacci_iter():

```

**Prompt:** Use a tool to get weather in SF. (Just describe the tool call you would make.)

```text
**Tool: “Weather‑SF‑Query”**  
**Command:** `weather_in_SF`  

This tool will fetch the current weather data for San Francis Bay (San Francis City) in the past‑day, next‑day, and next‑week. It will provide the latest temperature, precipitation, and air‑temperature (in Celsius) for the city, and the forecast for the next‑day and next‑week. The tool
```

**Prompt:** Prove that the sum of two even numbers is even.

```text
**Problem**  
Proprove that the sum of two even numbers is even.

**Solution**  
We can use the definition of even numbers.  
An even number is a number that is divisible by 2.  
Let \(a\) and \(b\) be two even numbers.  
Then \(a = 2k\) and \(b = 2l\) for some integers \(k\) and \(l\).  
The sum of two even
```


## Reproduce

```bash
modal run modal/structural_prune_smoke_20b.py
```
