"""Prompt templates for IKR (Intermediate Knowledge Representation) generation.

This module provides two-stage prompts for generating IKR:
1. Stage 1: Extract explicit facts, types, entities, and relations from the question
2. Stage 2: Generate background knowledge and rules given the explicit IKR

The two-stage approach:
- Reduces cognitive load per LLM call
- Makes background knowledge generation more targeted
- Improves accuracy on commonsense reasoning tasks
"""

# ==============================================================================
# Stage 1: Explicit Knowledge Extraction
# ==============================================================================

IKR_STAGE1_INSTRUCTIONS = '''You are a knowledge extraction system. Your task is to analyze a yes/no question and extract its logical structure into a formal representation.

## Output Format

You must output valid JSON matching this schema:

```json
{
  "meta": {
    "question": "<the original question>",
    "question_type": "yes_no"
  },
  "types": [
    {"name": "<TypeName>", "description": "<what this type represents>"}
  ],
  "entities": [
    {"name": "<entity_name>", "type": "<TypeName>", "aliases": ["<alternative names>"]}
  ],
  "relations": [
    {"name": "<relation_name>", "signature": ["<ArgType1>", "<ArgType2>"], "range": "Bool"}
  ],
  "facts": [
    {"predicate": "<relation_name>", "arguments": ["<entity1>", "<entity2>"], "negated": false, "source": "explicit"}
  ],
  "query": {
    "predicate": "<relation_name>",
    "arguments": ["<entity1>", "<entity2>"],
    "negated": false
  }
}
```

## Guidelines

### Types
- Create types for each category of entity mentioned (Person, Food, Animal, Location, etc.)
- Use PascalCase for type names (e.g., "Person", "FoodItem", "DietaryRestriction")
- Include a brief description of what the type represents

### Entities
- Extract all named individuals from the question
- Use snake_case for entity names (e.g., "plant_burger", "vegetarian_person")
- Include aliases that refer to the same entity in the question
- Each entity must have exactly one type

### Relations
- Create predicates for properties and relationships mentioned
- Use snake_case for relation names (e.g., "is_vegetarian", "would_eat", "contains")
- "signature" lists the argument types in order
- "range" is usually "Bool" for predicates, "Int" for numeric functions

### Facts
- Only include facts that are EXPLICITLY stated or clearly implied in the question
- Set "source" to "explicit" for all facts in Stage 1
- Use "negated": true for negative facts (e.g., "does not contain")

### Query
- This is what the question is asking about
- The query predicate and arguments must match a declared relation
- "negated": true if asking whether something is NOT the case

## Example

Question: "Would a vegetarian eat a burger made entirely of plants?"

```json
{
  "meta": {
    "question": "Would a vegetarian eat a burger made entirely of plants?",
    "question_type": "yes_no"
  },
  "types": [
    {"name": "Person", "description": "A human individual"},
    {"name": "Food", "description": "An edible item"}
  ],
  "entities": [
    {"name": "vegetarian_person", "type": "Person", "aliases": ["a vegetarian"]},
    {"name": "plant_burger", "type": "Food", "aliases": ["a burger made entirely of plants"]}
  ],
  "relations": [
    {"name": "is_vegetarian", "signature": ["Person"], "range": "Bool"},
    {"name": "is_plant_based", "signature": ["Food"], "range": "Bool"},
    {"name": "would_eat", "signature": ["Person", "Food"], "range": "Bool"}
  ],
  "facts": [
    {"predicate": "is_vegetarian", "arguments": ["vegetarian_person"], "negated": false, "source": "explicit"},
    {"predicate": "is_plant_based", "arguments": ["plant_burger"], "negated": false, "source": "explicit"}
  ],
  "query": {
    "predicate": "would_eat",
    "arguments": ["vegetarian_person", "plant_burger"],
    "negated": false
  }
}
```

## Important Notes

1. Only extract what is EXPLICITLY stated - do NOT add background knowledge yet
2. The query must be answerable with SAT (true) or UNSAT (false)
3. All entity names in facts and query must be declared in the entities list
4. All predicate names must be declared in the relations list
5. Wrap your JSON in ```json``` code blocks

'''


# ==============================================================================
# Stage 2: Background Knowledge Generation
# ==============================================================================

IKR_STAGE2_INSTRUCTIONS = '''You are a commonsense reasoning system. Given a partial knowledge representation of a question, your task is to add the BACKGROUND KNOWLEDGE (world knowledge, common sense) needed to answer the query.

## Current Knowledge Representation

The following types, entities, relations, and explicit facts have been extracted:

```json
{current_ikr}
```

## Your Task

Add background knowledge in the form of:
1. **Background facts**: General truths about the world
2. **Rules**: Universal implications that connect concepts

Output a JSON object with these new elements:

```json
{{
  "background_facts": [
    {{
      "predicate": "<relation_name>",
      "arguments": ["<entity_or_type>"],
      "negated": false,
      "source": "background",
      "justification": "<why this is common knowledge>"
    }}
  ],
  "rules": [
    {{
      "name": "<descriptive rule name>",
      "quantified_vars": [
        {{"name": "x", "type": "<TypeName>"}}
      ],
      "antecedent": {{
        "predicate": "<relation>",
        "arguments": ["x"]
      }},
      "consequent": {{
        "predicate": "<relation>",
        "arguments": ["x"]
      }},
      "justification": "<why this rule holds>"
    }}
  ]
}}
```

## Rule Structure

Rules encode universal implications: "For all x, if [antecedent] then [consequent]"

### Simple rule:
```json
{{
  "name": "vegetarians avoid meat",
  "quantified_vars": [{{"name": "p", "type": "Person"}}],
  "antecedent": {{"predicate": "is_vegetarian", "arguments": ["p"]}},
  "consequent": {{"predicate": "avoids_meat", "arguments": ["p"]}},
  "justification": "By definition, vegetarians do not eat meat"
}}
```

### Compound antecedent (AND):
```json
{{
  "name": "eating rule",
  "quantified_vars": [{{"name": "p", "type": "Person"}}, {{"name": "f", "type": "Food"}}],
  "antecedent": {{
    "and": [
      {{"predicate": "likes", "arguments": ["p", "f"]}},
      {{"predicate": "is_available", "arguments": ["f"]}}
    ]
  }},
  "consequent": {{"predicate": "would_eat", "arguments": ["p", "f"]}},
  "justification": "People eat food they like when it is available"
}}
```

### Negated consequent:
```json
{{
  "consequent": {{"predicate": "contains_meat", "arguments": ["f"], "negated": true}}
}}
```

## Guidelines

1. **Be minimal**: Only add knowledge NECESSARY to connect explicit facts to the query
2. **Be general**: Rules should be universally true, not specific to this question
3. **Chain reasoning**: Ensure rules form a logical chain from facts to query
4. **Justify everything**: Each piece of background knowledge needs a justification

## Reasoning Process

1. Look at the query: what predicate needs to be established?
2. Look at the explicit facts: what do we know?
3. Identify the GAP: what knowledge bridges facts to query?
4. Add minimal rules/facts to fill that gap

## Example

Given explicit facts about a vegetarian and plant burger, to answer "would the vegetarian eat it?":

Gap analysis:
- We know: is_vegetarian(person), is_plant_based(burger)
- We need: would_eat(person, burger)
- Missing: What determines if someone would eat something?

Background knowledge needed:
1. Vegetarians avoid meat (definition)
2. Plant-based foods contain no meat (definition)
3. People eat food that doesn't violate their dietary restrictions (common sense)

```json
{{
  "background_facts": [],
  "rules": [
    {{
      "name": "vegetarians avoid meat",
      "quantified_vars": [{{"name": "p", "type": "Person"}}],
      "antecedent": {{"predicate": "is_vegetarian", "arguments": ["p"]}},
      "consequent": {{"predicate": "avoids_meat", "arguments": ["p"]}},
      "justification": "Definition of vegetarian diet"
    }},
    {{
      "name": "plant-based means no meat",
      "quantified_vars": [{{"name": "f", "type": "Food"}}],
      "antecedent": {{"predicate": "is_plant_based", "arguments": ["f"]}},
      "consequent": {{"predicate": "contains_meat", "arguments": ["f"], "negated": true}},
      "justification": "Plant-based foods by definition contain no meat"
    }},
    {{
      "name": "dietary compatibility",
      "quantified_vars": [{{"name": "p", "type": "Person"}}, {{"name": "f", "type": "Food"}}],
      "antecedent": {{
        "and": [
          {{"predicate": "avoids_meat", "arguments": ["p"]}},
          {{"predicate": "contains_meat", "arguments": ["f"], "negated": true}}
        ]
      }},
      "consequent": {{"predicate": "would_eat", "arguments": ["p", "f"]}},
      "justification": "People eat foods compatible with their dietary restrictions"
    }}
  ]
}}
```

Wrap your JSON in ```json``` code blocks.
'''


def build_ikr_stage1_prompt(question: str) -> str:
    """Build the Stage 1 prompt for explicit knowledge extraction.

    Args:
        question: The natural language question

    Returns:
        Complete prompt for Stage 1
    """
    return IKR_STAGE1_INSTRUCTIONS + f"\n\nQuestion: {question}"


def build_ikr_stage2_prompt(current_ikr: str) -> str:
    """Build the Stage 2 prompt for background knowledge generation.

    Args:
        current_ikr: JSON string of the current IKR (from Stage 1)

    Returns:
        Complete prompt for Stage 2
    """
    return IKR_STAGE2_INSTRUCTIONS.format(current_ikr=current_ikr)


# ==============================================================================
# Combined Single-Stage Prompt (Alternative)
# ==============================================================================

IKR_SINGLE_STAGE_INSTRUCTIONS = '''You are a logical reasoning system. Your task is to analyze a yes/no question and create a complete formal representation including:
1. Explicit facts from the question
2. Background knowledge needed to answer it
3. Logical rules connecting facts to the answer

## Output Format

Output valid JSON matching this schema:

```json
{
  "meta": {
    "question": "<the original question>",
    "question_type": "yes_no"
  },
  "types": [
    {"name": "<TypeName>", "description": "<what this type represents>"}
  ],
  "entities": [
    {"name": "<entity_name>", "type": "<TypeName>"}
  ],
  "relations": [
    {"name": "<relation_name>", "signature": ["<Type1>"], "range": "Bool"}
  ],
  "facts": [
    {"predicate": "<name>", "arguments": ["<entity>"], "source": "explicit"},
    {"predicate": "<name>", "arguments": ["<entity>"], "source": "background", "justification": "<why>"}
  ],
  "rules": [
    {
      "name": "<rule name>",
      "quantified_vars": [{"name": "x", "type": "<Type>"}],
      "antecedent": {"predicate": "<name>", "arguments": ["x"]},
      "consequent": {"predicate": "<name>", "arguments": ["x"]},
      "justification": "<why this rule holds>"
    }
  ],
  "query": {
    "predicate": "<relation_name>",
    "arguments": ["<entity>"]
  }
}
```

## Guidelines

1. **Types**: Categories of entities (Person, Food, Location, etc.)
2. **Entities**: Named individuals from the question
3. **Relations**: Predicates and functions connecting entities
4. **Explicit facts**: Directly stated in the question (source: "explicit")
5. **Background facts**: World knowledge needed (source: "background")
6. **Rules**: Universal implications (forall x: antecedent(x) => consequent(x))
7. **Query**: What the question asks about

## Reasoning Strategy

1. Extract explicit facts from the question
2. Identify what the query is asking
3. Determine the MINIMAL background knowledge needed to connect facts to query
4. Express that knowledge as rules with justifications

Wrap your JSON in ```json``` code blocks.
'''


def build_ikr_single_stage_prompt(question: str) -> str:
    """Build a single-stage prompt for complete IKR generation.

    This is an alternative to two-stage prompting, useful for
    simpler questions or when minimizing LLM calls.

    Args:
        question: The natural language question

    Returns:
        Complete single-stage prompt
    """
    return IKR_SINGLE_STAGE_INSTRUCTIONS + f"\n\nQuestion: {question}"
