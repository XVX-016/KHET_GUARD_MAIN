# Outputs for Khet Guard Infrastructure

output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "private_subnets" {
  description = "List of IDs of private subnets"
  value       = module.vpc.private_subnets
}

output "public_subnets" {
  description = "List of IDs of public subnets"
  value       = module.vpc.public_subnets
}

output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "Security group ids attached to the cluster control plane"
  value       = module.eks.cluster_security_group_id
}

output "cluster_iam_role_name" {
  description = "IAM role name associated with EKS cluster"
  value       = module.eks.cluster_iam_role_name
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
}

output "cluster_oidc_issuer_url" {
  description = "The URL on the EKS cluster for the OpenID Connect identity provider"
  value       = module.eks.cluster_oidc_issuer_url
}

output "cluster_oidc_provider_arn" {
  description = "The ARN of the OIDC Provider if EKS IRSA is enabled"
  value       = module.eks.oidc_provider_arn
}

output "db_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.khet_guard_db.endpoint
}

output "db_port" {
  description = "RDS instance port"
  value       = aws_db_instance.khet_guard_db.port
}

output "db_name" {
  description = "RDS instance database name"
  value       = aws_db_instance.khet_guard_db.db_name
}

output "redis_endpoint" {
  description = "Redis cluster primary endpoint"
  value       = aws_elasticache_replication_group.khet_guard.primary_endpoint_address
}

output "redis_port" {
  description = "Redis cluster port"
  value       = aws_elasticache_replication_group.khet_guard.port
}

output "models_bucket_name" {
  description = "S3 bucket name for model artifacts"
  value       = aws_s3_bucket.khet_guard_models.bucket
}

output "models_bucket_arn" {
  description = "S3 bucket ARN for model artifacts"
  value       = aws_s3_bucket.khet_guard_models.arn
}

output "data_bucket_name" {
  description = "S3 bucket name for data storage"
  value       = aws_s3_bucket.khet_guard_data.bucket
}

output "data_bucket_arn" {
  description = "S3 bucket ARN for data storage"
  value       = aws_s3_bucket.khet_guard_data.arn
}

output "service_role_arn" {
  description = "ARN of the IAM role for Khet Guard services"
  value       = aws_iam_role.khet_guard_service_role.arn
}

output "kubeconfig_command" {
  description = "Command to configure kubectl for the EKS cluster"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${var.cluster_name}"
}
