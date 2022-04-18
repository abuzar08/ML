
def InitializeBeams(y_probs):
    '''
    DESC
    ---
    Initializes beam search

    ---
    ARGS
    ---
    y_probs: (vocab_size) -> log_probs at time 0

    ---
    RETURNS
    ---
    candidate_paths: dict()
    final_paths: dict()
    '''
    path = "<sos>"
    candidate_paths = {}
    final_paths = {}
    candidate_paths[path] = y_probs[0]

    return candidate_paths, final_paths

def update_beams(y_probs, path_scores):
    '''
    DESC
    ---
    Looks ahead one time-step
    
    ---
    ARGS
    ---
    y_probs: (vocab_size) -> log_probs at time t

    ---
    RETURNS
    ---
    updated_path_scores: dict()
    '''
    current_paths = set(path_scores.keys())
    updated_path_scores = path_scores.copy()
    for path in current_paths:
        for c in LETTER_LIST[1:]:
            new_path = path + c
            new_score = path_scores[path] + y_probs[letter2index[c]]
            updated_path_scores[new_path] = new_score
    return updated_path_scores

def prune_beams(path_scores, beam_width):
    '''
    DESC
    ---
    Prunes current search space to the best <beam_width> paths

    ---
    ARGS
    ---
    path_scores: dict()
    beam_width; int

    ---
    RETURNS
    ---
    updated_path_scores: dict()
    path_scores_with_eos: dict()
    updated_beam_width: dict()
    '''
    candidate_paths = list(path_scores.keys())
    candidate_paths = sorted(candidate_paths, key= lambda x: path_scores[x], reverse=True)

    if len(candidate_paths) > beam_width:
        candidate_paths = candidate_paths[:beam_width]
    
    candidate_paths_with_eos = [path for path in candidate_paths if path[-5:]=="<eos>"]
    if len(candidate_paths_with_eos) == 0:
        updated_path_scores = {path:path_scores[path] for path in candidate_paths}
        return updated_path_scores, dict(), beam_width
    
    path_scores_with_eos = {path:path_scores[path] for path in candidate_paths_with_eos}
    updated_path_scores = {path:path_scores[path] for path in candidate_paths if path not in path_scores_with_eos}
    updated_beam_width = beam_width - len(path_scores_with_eos)

    return updated_path_scores, path_scores_with_eos, updated_beam_width

def update_final_beams(final_paths, path_scores_with_eos):
    updated_final_paths = final_paths.copy()
    for path, score in path_scores_with_eos.items():
        updated_final_paths[path] = score
    return updated_final_paths

class BearchDecoder:
    def __init__(self, beam_width):
        self.beam_width = beam_width
    
    def decode(self, y):
        '''
        DESC
        ---
        Returns best path for ONE sequence
        
        ARGS
        ---
        y: (T,Vocab_Size)

        RETURN
        ---
        best_path: string()
        
        '''
        path_scores, final_paths = InitializeBeams(y[0])
        T, vocab_size = y.shape
        for t in range(1, T):
            y_probs = torch.log(y[t])
            paths_scores = update_beams(y_probs)
            path_scores, path_scores_with_eos, self.beam_width = prune_beams(path_scores, self.beam_width)
            final_paths = update_final_beams(final_paths, path_scores_with_eos)

            if self.beam_width == 0:
                break
            
            if t == T-1:
                for path in path_scores:
                    final_paths[path] = path_scores[path]
        
        candidate_paths = list(final_paths.keys())
        best_path = sorted(candidate_paths, key= lambda x: final_paths[x], reverse=True)

        return best_path
