
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select ticker
from "stock_intelligence"."main"."mart_daily_returns"
where ticker is null



  
  
      
    ) dbt_internal_test