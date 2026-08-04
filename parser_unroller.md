Definition 1. SCCInfo - combined loop structure information with corresponding header stacks
Definition 2. StackAccess - description of header stack used in parser state
Definition 3. PendingSCC - a loop candidate to be rejected or unrolled

Stage 1. Find loop components based on how many back edges exist. (findBackEdges)
Stage 2. For each state in parser gather stack accesses. (computeStackAccesses)
Stage 3. Build per-loop SCCInfo (buildSCCInfo)
Substage 3.1 Gather positions of parser states(computedeclarationPositions) - remove this stage
Substage 3.2 Convert backEdges into loop candidates, group backedges by their destination, create PendingSCC for each (collectPendingSCCs)
Substage 3.3 Validate loop candidates(reject too deep/no header stack/untrackable header stack) and compute combined loop information[combinedSCCAccesses] (acceptLoopSCC)
Substage 3.4 Determine which header stacks will be keys for each state(computeRelevantStacks)
Stage 4. Calculate loop iteration values/header stack indexes. (runSymbolicExecution)
Stage 5. Create unrolled stages using gathered information. (materializeUnrolled)
Substage 5.0 Error out if no reject state found or if unrolling reached out of bounds.
Substage 5.1 Create a clone for each new unrolled state(createClones)
Substage 5.2 Replace constant indices with new values(substituteConstantIndices)
Substage 5.3 Replace foldable expression indices (substituteExplicitIndices)
Substage 5.4 Update target states of transitions in clones(rewriteTransition)
