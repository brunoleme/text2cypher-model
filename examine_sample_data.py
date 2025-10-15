#!/usr/bin/env python3
"""
Quick script to examine the sample dataset structure
"""

import pandas as pd
import os

def examine_sample_data():
    sample_path = "src/text2cypher/finetuning/data/text2cypher_sample.parquet"
    
    if not os.path.exists(sample_path):
        print(f"❌ Sample file not found: {sample_path}")
        return
    
    print(f"📊 Examining sample dataset: {sample_path}")
    
    try:
        # Load the parquet file
        df = pd.read_parquet(sample_path)
        
        print(f"\n📏 Dataset shape: {df.shape} (rows, columns)")
        print(f"\n📝 Columns: {list(df.columns)}")
        print(f"\n🔍 Column types:")
        for col in df.columns:
            print(f"  - {col}: {df[col].dtype}")
        
        print(f"\n📋 Sample records:")
        for i, row in df.head(3).iterrows():
            print(f"\n--- Record {i+1} ---")
            for col in df.columns:
                value = str(row[col])
                if len(value) > 200:
                    value = value[:200] + "..."
                print(f"{col}: {value}")
        
        # Check for missing values
        print(f"\n❌ Missing values per column:")
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            print(f"  - {col}: {missing_count} ({missing_pct:.1f}%)")
        
        # Basic statistics
        if 'question' in df.columns:
            question_lengths = df['question'].str.len()
            print(f"\n📏 Question character lengths:")
            print(f"  - Min: {question_lengths.min()}")
            print(f"  - Max: {question_lengths.max()}")
            print(f"  - Mean: {question_lengths.mean():.1f}")
            print(f"  - Median: {question_lengths.median():.1f}")
        
        if 'cypher' in df.columns:
            cypher_lengths = df['cypher'].str.len()
            print(f"\n📏 Cypher query character lengths:")
            print(f"  - Min: {cypher_lengths.min()}")
            print(f"  - Max: {cypher_lengths.max()}")
            print(f"  - Mean: {cypher_lengths.mean():.1f}")
            print(f"  - Median: {cypher_lengths.median():.1f}")
        
        print(f"\n✅ Sample data examination complete!")
        
    except Exception as e:
        print(f"❌ Error examining sample data: {e}")

if __name__ == "__main__":
    examine_sample_data()

