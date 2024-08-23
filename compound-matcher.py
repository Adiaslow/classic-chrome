class CompoundMatcher:
    """Class for matching and optimizing compounds."""

    def __init__(self, n_1: str = 'AgxNull'):
        self.n_1 = n_1

    def create_compound_key(self, row: pd.Series) -> Tuple[str, str, str]:
        """Create a compound key from a row."""
        return tuple(row[f'BB{i} Name'] if row[f'BB{i} Name'] != self.n_1 else self.n_1 for i in range(1, 4))

    def optimize_compound_matching(self, df: pd.DataFrame, building_blocks: List[str]) -> pd.DataFrame:
        """Optimize compound matching."""
        full_perm = tuple(building_blocks)
        all_perms = self._generate_all_permutations(building_blocks)

        df['compound_key'] = df.apply(self.create_compound_key, axis=1)
        matched_df = df[df['compound_key'].isin(all_perms)].copy()

        compound_groups = self._create_compound_groups(building_blocks, full_perm, all_perms)
        matched_df['group'] = matched_df['compound_key'].map(compound_groups)

        return matched_df

    def _generate_all_permutations(self, building_blocks: List[str]) -> Set[Tuple[str, str, str]]:
        """Generate all possible permutations including truncations."""
        all_perms = set()

        # Full compound
        all_perms.add(tuple(building_blocks))

        # Single truncations
        for i in range(3):
            truncation = list(building_blocks)
            truncation[i] = self.n_1
            all_perms.add(tuple(truncation))

        # Double truncations
        for i in range(3):
            truncation = [self.n_1, self.n_1, self.n_1]
            truncation[i] = building_blocks[i]
            all_perms.add(tuple(truncation))

        # Fully null compound
        all_perms.add((self.n_1, self.n_1, self.n_1))

        return all_perms

    def _create_compound_groups(self, building_blocks: List[str], full_perm: Tuple[str, str, str],
                                all_perms: Set[Tuple[str, str, str]]) -> Dict[Tuple[str, str, str], str]:
        """Create compound groups."""
        compound_groups = {
            full_perm: '_'.join(building_blocks),
            (self.n_1, self.n_1, self.n_1): 'Null'
        }

        for perm in all_perms:
            if perm == full_perm or perm == (self.n_1, self.n_1, self.n_1):
                continue
            non_nulls = [bb for bb in perm if bb != self.n_1]
            if len(non_nulls) == 1:
                compound_groups[perm] = non_nulls[0]
            elif len(non_nulls) == 2:
                compound_groups[perm] = f"{non_nulls[0]}_{non_nulls[1]}"

        return compound_groups

    def analyze_compound_truncations(self, df: pd.DataFrame, compound: Tuple[str, str, str]) -> Dict:
        """Analyzes truncations of a given compound."""
        results = {}

        # Get RT value for the compound
        compound_data = df[df['compound_key'] == compound]
        if compound_data.empty:
            return None  # Skip compounds with no data

        compound_rt = compound_data['Cyclized RT (min)'].values[0]
        results['full'] = {'rt': compound_rt}

        # Get RT value for the fully masked compound
        masked_compound = (self.n_1, self.n_1, self.n_1)
        masked_data = df[df['compound_key'] == masked_compound]
        if masked_data.empty:
            return None  # Skip if no fully masked compound data

        masked_rt = masked_data['Cyclized RT (min)'].values[0]
        results['masked'] = {'rt': masked_rt}

        # Generate and analyze all possible truncations
        truncations = self._generate_truncations(compound)
        for truncated_compound in truncations:
            truncated_data = df[df['compound_key'] == truncated_compound]
            if not truncated_data.empty:
                rts = truncated_data['Cyclized RT (min)'].values
                results[truncated_compound] = {
                    'mean': np.mean(rts),
                    'std': np.std(rts),
                    'diff_full': np.mean(rts) - compound_rt,
                    'diff_masked': np.mean(rts) - masked_rt,
                    'rts': rts.tolist()
                }

        return results

    def _generate_truncations(self, compound: Tuple[str, str, str]) -> List[Tuple[str, str, str]]:
        """Generate all possible truncations of a compound."""
        truncations = [compound]
        for i in range(1, 3):
            truncations.extend(itertools.combinations(compound, i))
        return [t if len(t) == 3 else t + (self.n_1,) * (3 - len(t)) for t in truncations]

    def print_truncation_analysis(self, df: pd.DataFrame, x: str, y: str, z: str):
        """Prints the truncation analysis for a specific compound."""
        compound = (x, y, z)
        results = self.analyze_compound_truncations(df, compound)
        if results is None:
            print(f"No data available for compound: {x}-{y}-{z}")
            return

        full_data = results['full']
        masked_data = results['masked']
        print(f"Full Compound: {x}-{y}-{z}")
        print(f"  RT: {full_data['rt']:.4f}")
        print()

        print("Truncation Analysis:")
        for truncated_compound, data in results.items():
            if truncated_compound in ['full', 'masked']:
                continue
            compound_str = '-'.join(bb if bb != self.n_1 else 'X' for bb in truncated_compound)
            print(f"  Truncated Compound: {compound_str}")
            print(f"    Mean RT: {data['mean']:.4f}")
            print(f"    Std RT: {data['std']:.4f}")
            print(f"    Difference from full compound RT: {data['diff_full']:.4f}")
            print(f"    Difference from true null RT: {data['diff_masked']:.4f}")
            print(f"    RTs: {', '.join([f'{rt:.4f}' for rt in data['rts']])}")
            print()