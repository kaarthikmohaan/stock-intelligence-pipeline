
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select avg_daily_return_pct
from "stock_intelligence"."main"."mart_daily_returns"
where avg_daily_return_pct is null



  
  
      
    ) dbt_internal_test