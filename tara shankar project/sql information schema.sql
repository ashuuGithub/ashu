USE msdb;
GO

SELECT 
    ROUTINE_SCHEMA AS SchemaName,
    ROUTINE_NAME AS ProcedureName,
 
FROM 
    INFORMATION_SCHEMA.ROUTINES
WHERE 
    ROUTINE_SCHEMA = 'dbo'
    AND ROUTINE_NAME like '%YourProcedureName%'
    AND ROUTINE_TYPE = 'PROCEDURE';









