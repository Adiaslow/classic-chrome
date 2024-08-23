class FPADataPreprocessor:
    def __init__(self, preprocessed_df: pd.DataFrame, n_1: str = 'AgxNull'):
        self.preprocessed_df = preprocessed_df
        self.n_1 = n_1
        self.building_blocks = self._extract_building_blocks()
        self.full_compound_dict = self._create_full_compound_dict()
        self.truncated_compound_dict = self._create_truncated_compound_dict()

    def _extract_building_blocks(self) -> set:
        bb_columns = ['BB1 Name', 'BB2 Name', 'BB3 Name']
        building_blocks = set(self.preprocessed_df[bb_columns].values.ravel())
        building_blocks.discard(self.n_1)
        return building_blocks

    def _create_full_compound_dict(self) -> Dict[Tuple[str, ...], List[Tuple[Tuple[str, ...], float, Dict]]]:
        data = self.preprocessed_df[['SMILES', 'BB1 Name', 'BB2 Name', 'BB3 Name', 'Cyclized RT (min)', 'Times', 'Counts', 'Deduplicated Counts', 'Success']].values

        full_dict = defaultdict(list)
        for row in data:
            compound = tuple(bb if bb != self.n_1 else self.n_1 for bb in row[1:4])  # Use BB1, BB2, BB3 for the compound key
            try:
                chromatogram = {
                    'times': row[5],
                    'counts': row[6],
                    'deduplicated_counts': row[7]
                }
                full_dict[compound].append((compound, row[4], chromatogram))
            except Exception as e:
                print(f"Error processing compound {compound}: {str(e)}")
                print(f"Row data: {row}")

        return dict(full_dict)

    def _create_truncated_compound_dict(self) -> Dict[Tuple[str, ...], List[Tuple[Tuple[str, ...], float, Dict]]]:
        truncated_dict = defaultdict(list)
        for compound, data_list in self.full_compound_dict.items():
            non_null_bbs = tuple(bb for bb in compound if bb != self.n_1)
            if len(non_null_bbs) < 3:
                truncated_dict[non_null_bbs].extend(data_list)
        return dict(truncated_dict)


    def check_data_integrity(self):
        print(f"Total rows in preprocessed data: {len(self.preprocessed_df)}")
        print(f"Rows with valid 'Times' data: {self.preprocessed_df['Times'].notna().sum()}")
        print(f"Rows with valid 'Counts' data: {self.preprocessed_df['Counts'].notna().sum()}")
        print(f"Rows with valid 'Deduplicated Counts' data: {self.preprocessed_df['Deduplicated Counts'].notna().sum()}")

        sample_row = self.preprocessed_df.iloc[0]
        print("\nSample row:")
        print(f"BB1 Name: {sample_row['BB1 Name']}")
        print(f"BB2 Name: {sample_row['BB2 Name']}")
        print(f"BB3 Name: {sample_row['BB3 Name']}")
        print(f"Times: {sample_row['Times'][:10]}...")  # Print first 100 characters
        print(f"Counts: {sample_row['Counts'][:10]}...")
        print(f"Deduplicated Counts: {sample_row['Deduplicated Counts'][:10]}...")

    def check_full_compound_dict(self):
        print("\nChecking full_compound_dict:")
        print(f"Total number of compounds: {len(self.full_compound_dict)}")
        print("Sample of compounds:")
        for i, (key, value) in enumerate(list(self.full_compound_dict.items())[:5]):
            print(f"Compound {i+1}: {key}")
            print(f"  Number of chromatograms: {len(value)}")
            if value:
                print(f"  Sample chromatogram keys: {list(value[0][2].keys())}")
            print()

        # Check for the specific compound we're analyzing
        specific_compound = ('Leu', 'Phe', 'Val')
        if specific_compound in self.full_compound_dict:
            print(f"Data for {specific_compound}:")
            print(f"  Number of chromatograms: {len(self.full_compound_dict[specific_compound])}")
            if self.full_compound_dict[specific_compound]:
                print(f"  Sample chromatogram: {self.full_compound_dict[specific_compound][0][2]}")
        else:
            print(f"No data found for {specific_compound}")

        # Check for double truncations
        for bb in ['Leu', 'Phe', 'Val']:
            double_truncation = (bb, self.n_1, self.n_1)
            if double_truncation in self.full_compound_dict:
                print(f"Data found for double truncation {bb}")
                print(f"  Number of chromatograms: {len(self.full_compound_dict[double_truncation])}")
            else:
                print(f"No data found for double truncation {bb}")