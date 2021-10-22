# HouseSales-RecommendationSystem

This project is a recommendation system for Real Estate companies based on insights from exploratory data analysis.

1. **Business Understanding**
    
    Domain: Real Estate
    
    Business: Buy houses for a price value and Sell them for a higher price 
    
2. **Business Problems (presented by business experts)**
    1. Which houses should be bought and for what price?
    2. Once its bought when it's the best time period to sell it and for what price?
3. **Solution Proposal**
    
    
    1. Build a table with house recommendations to buy or not to buy.
    
    1. Final deliverables:
        1. A report with suggestions of houses to buy along with the price.
        2. In PDF.
    2. Expected Tools:
        1. Python 3.8.0
        2. Jupyter Notebook
    3. Tasks process:
        
        First we will define the proposal tasks and its guidelines to answer the business problems presented:
        
        1. Which houses should be bought and for what price?
            
            Planning proposal 1:
            
            - With all the data collected, integrated and treated,
            - Create a new dataframe with median variable for each region, and the zipcode, by grouping the data by zipcode, and calculate the median for each sub-group,
            - Then merge the original dataframe with the new one, on zipcode and with a left outer join, or inner join, to return all the rows that match with the first dataframe and add the correspondent median,
            - Then rise some conditional hypothesis:
                - The houses that have a price value lower than the median and are in good conditions, can be sold for a higher price, so are good to buy.
                - The houses that have a price value lower than the median and are in bad conditions, cannot be sold for a higher price, so are not good to buy.
                - The houses that have a price value higher than the median, independently from the condition, are not good to buy and take profit.
            - Create a new variable that indicates what should be bought or not
            
            Data sample 1:
            
            house id | zipcode | house price | median price | condition | Status
            
            185074   | 385421 | 450 000       | 500 000        | 3              | to buy
            
            145879   | 785963 | 400 000       | 500 000        | 1              | don't buy
            
            145879   | 785963 | 750 000       | 500 000        | 2              | don't buy
            
            Code sample 1:
            
            ```python
            df = orig_df[['zipcode','price']].groupby('zipcode').median().reset_index()
            df.columns = ['zipcode', 'median_price']
            merged_df = pd.merge(orig_df,df,on='zipcode',how='inner')
            
            for i in range(len(merged_df)):
            	if (merged_df[i,'price'] < merged_df[i,'median_price']) &\
            		 (merged_df[i,'condition'] >= 2):
            		merged_df['status'] = 'to buy'
            	else:
            		merged_df['status'] = 'don\'t buy'
            ```
            
    
    2. Build a table with best time periods recommendations to sell houses and for which prices.
    
    1. Final deliverables:
        1. A report with suggestions of time periods to sell those houses along with the price.
        2. In PDF.
    2. Expected Tools:
        1. Python 3.8.0
        2. Jupyter Notebook
    3. Tasks process:
        
        Planning Proposal 1:
        
        - Assuming that the data was already collected, integrated and treated. And the variable about 'selling_price' is available. But it depends on what attributes are available, regarding the date of purchase the acquiring price, and some records with both acquiring and sold price.
        - First define a new variable, 'season', that is representing seasonal quarters, ['Spring', 'Summer', 'Autumn', 'Winter'], extracted from 'date' according to local data.
        - Then group the dataframe by 'zipcode' and sub-group by 'season', and calculate the median selling price per season per zipcode.
        - Then calculate the maximum median value for each zipcode sub-group, and get the seasonal quarter which has the highest median selling price.
        - Then merge it to the original dataframe, on zipcode and with a left or inner join. And now we have answered which is the best time to sell, assuming that the reason the seasonal quarter had the highest median by chance, because there could be a random chance that in that time period some higher price value houses were all sold.
        - And create some conditional hypothesis:
            - The houses selling price will be equal to the median + 10% if the house characteristics are better, like condition = 3.
            - The houses selling price will be equal to the median - 10% if the house characteristics are worse, like condition = 3.
        
        Data sample 1:
        
        house id | zipcode | house price | best_selling_quarter | median_selling_price | condition | selling_price
        
        10330     | 857496 | 450 000       | Summer                   | 700 000                     | 3             |  700 000 + 10%
        
        Code sample 1:
        
        Planning Proposal 2:
        
        - Same as previous one but now adding the number of bedrooms and sqft_lot to the seasonal aggregation to calculate a median for houses with the same characteristics.
        - And the percentage of the median selling price will be based on the YoY (Year over Year) variation. If its positive plus 10%, much higher than previous years plus 20%, etc.
    
    3. **Create visualizations to answer to each one of the 10 business hypothesis (made by Data Scientists).**
    
    - H1: Houses that have a water view, are 20% more expensive on average.
    - H2: Houses with year built older than 1955, are 50% cheaper on average.
    - H3: Houses without basement are 40% bigger than with basement.
    - H4: The growth rate of the houses price YoY (Year over Year) is 10%.
    - H5: Houses with 3 bathrooms have a MoM (Month over Month) growth of 15%.
    - H6:
    
    4. **Provide an interactive and web accessible Dashboard with the business problems answered and the information to evaluate the proposed hypothesis and extract insights.**
    
    1. Final deliverables:
        1. An URL for the access to the dashboard, where the business insights and results will appear.
    2.  Expected Tools:
        1. Pycharm
        2. Streamlit
        3. Heroku
    3. Tasks process:
        1. Create Procfile, setup.sh, requirements.txt etc.
        
4. **Presentation of the 5 main insights (answering some expert questions and other hypothesis)**
    1. Insights visualization and interpretation
    
5. **Business Results**
    1. Actionable solutions 
    2. Business metrics comparison between old and newer actions.
6. **Conclusion**
    1. Was the initial goal achieved?
7. **Next actionable steps**
