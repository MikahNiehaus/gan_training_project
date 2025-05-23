"""
Fixed training loop without indentation issues for training_gan.py
This is a version of the train method without any indentation problems
"""

def train(self, train_loader, epochs=float('inf'), sample_interval=1000):
    """Fixed training loop to resolve indentation issues and improve training"""
    local_train_loader = train_loader
    self.load_checkpoint()
    epoch = self.start_epoch
    best_val = float('inf')
    no_improve = 0
    prev_G_loss = None
    vram_retry = 0
    max_epochs = epochs
    self.epochs_at_res = 0  # Track epochs at current resolution
    manual_distortion_scale = 0.8 # Set your fixed distortion value here
    distortion = manual_distortion_scale if self.use_distortion else 0.0  # Use full value, no cap
    
    while epoch < max_epochs:
        # Set a new random seed for each epoch for better randomness
        epoch_seed = int(time.time()) + epoch
        random.seed(epoch_seed)
        torch.manual_seed(epoch_seed)
        np.random.seed(epoch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(epoch_seed)
        
        try:
            # Enable anomaly detection temporarily to identify the exact operation causing the issue
            torch.autograd.set_detect_anomaly(True)
            D_losses = []
            G_losses = []
            D_paused_batches = 0
            G_paused_batches = 0
            D_paused_streak = 0
            G_paused_streak = 0
            max_D_paused_streak = 0
            max_G_paused_streak = 0
            last_paused = None
            epoch_start = time.time()
            pbar = tqdm(local_train_loader, desc=f"Epoch {epoch+1} [{self.img_size}x{self.img_size}]", leave=False)
            use_amp = hasattr(self, 'use_amp') and self.use_amp
            batch_times = []
            data_times = []
            batch_start = time.time()
            distortion_values = []  # Track distortion values for this epoch (for logging only)
            example_distortion_saved = False
            
            # Process each batch in the epoch
            for real in pbar:
                data_loaded = time.time()
                data_times.append(data_loaded - batch_start)
                real = real.to(self.device)
                batch_size = real.size(0)
                noise = torch.randn(batch_size, self.z_dim, 1, 1, device=self.device)
                
                # Set fixed learning rates
                for param_group in self.opt_G.param_groups:
                    param_group['lr'] = 0.0002  # Slightly higher LR for generator
                for param_group in self.opt_D.param_groups:
                    param_group['lr'] = 0.0001  # Lower LR for discriminator to slow it down
                
                # 1. Train Discriminator: train less frequently than generator
                # Only update discriminator every other batch to prevent it from overpowering the generator
                train_discriminator = True
                if epoch > 0 and avg_D_loss is not None and avg_G_loss is not None:
                    if avg_D_loss < 0.3 and avg_D_loss < avg_G_loss / 2:
                        # If discriminator is too strong, skip training it this batch
                        train_discriminator = random.random() > 0.5
                
                if train_discriminator:
                    self.opt_D.zero_grad()
                    
                    # Generate fake images
                    with torch.no_grad():  # Don't track gradients here to save memory
                        fake_images = self.G(noise)
                    
                    # Real images loss - add noise to labels for smoother training
                    D_real = self.D(real).view(-1)
                    # Use soft real labels between 0.7 and 1.0 rather than exactly 1
                    real_label = torch.full_like(D_real, 0.9)
                    if self.use_distortion:
                        real_label = real_label - distortion * torch.rand_like(D_real) * 0.2  # smaller noise
                    loss_D_real = self.criterion(D_real, real_label)
                    
                    # Fake images loss - add noise to labels for smoother training
                    D_fake = self.D(fake_images.detach()).view(-1)
                    # Use soft fake labels between 0.0 and 0.3 rather than exactly 0
                    fake_label = torch.full_like(D_fake, 0.1)
                    if self.use_distortion:
                        fake_label = fake_label + distortion * torch.rand_like(D_fake) * 0.2  # smaller noise
                    loss_D_fake = self.criterion(D_fake, fake_label)
                    
                    # Combined discriminator loss
                    loss_D = loss_D_real + loss_D_fake
                    loss_D.backward()
                    self.opt_D.step()
                else:
                    loss_D = torch.tensor(0.0, device=self.device)
                    D_paused_batches += 1
                
                # 2. Train Generator: always train the generator
                self.opt_G.zero_grad()
                
                # Generate new fake images for generator update (don't reuse)
                fake_for_G = self.G(noise)
                
                # Feed fake images into discriminator
                D_output = self.D(fake_for_G).view(-1)
                
                # Generator wants discriminator to classify fake images as real
                # Target for generator is always 1 (real)
                target_real = torch.full_like(D_output, 0.9)  # Use 0.9 instead of 1.0 for smoother training
                loss_G = self.criterion(D_output, target_real)
                
                # Feature matching loss - helps generator match statistics of real data
                if isinstance(fake_for_G, torch.Tensor) and isinstance(real, torch.Tensor) and fake_for_G.size() == real.size():
                    # Calculate feature matching loss: make fake images statistically similar to real ones
                    feature_match_loss = torch.mean(torch.abs(
                        torch.mean(fake_for_G, dim=0) - torch.mean(real, dim=0)
                    ))
                    loss_G = loss_G + 0.1 * feature_match_loss  # Add small weight to feature matching
                    
                    # Use fixed distortion for all batches
                    # Only save example-distortion.png once per epoch
                    if not example_distortion_saved:
                        import os
                        from torchvision.utils import save_image
                        real_img = real[0].detach().cpu()
                        if self.use_distortion and distortion > 0.0:
                            # Add random noise proportional to distortion
                            noise = torch.randn_like(real_img) * distortion
                            distorted_img = real_img + noise
                            distorted_img = torch.clamp(distorted_img, -1.0, 1.0)
                            save_image(distorted_img, os.path.join(SAMPLE_DIR, 'example-distortion.png'), normalize=True)
                            print(f"[DISTORTION] Saved example-distortion.png with distortion={distortion:.4f} (noise)")
                        else:
                            save_image(real_img, os.path.join(SAMPLE_DIR, 'example-distortion.png'), normalize=True)
                            print(f"[DISTORTION] Saved example-distortion.png with distortion=0 (no distortion)")
                        example_distortion_saved = True
                    
                    # Remove per-batch distortion log; accumulate for epoch
                    if 'distortion_accum' not in locals():
                        distortion_accum = 0.0
                        distortion_batches = 0
                    distortion_accum += distortion
                    distortion_batches += 1
                    
                # Neighbor penalty: encourage diverse outputs
                neighbor_penalty_lambda = 0.5 if self.use_neighbor_penalty else 0.0
                neighbor_penalty = 0.0
                
                if self.use_neighbor_penalty and batch_size > 1:
                    # Calculate pairwise differences between generated images
                    # This encourages the generator to produce diverse images
                    fake_flat = fake_for_G.view(batch_size, -1)
                    similarity_matrix = torch.mm(fake_flat, fake_flat.t())
                    
                    # Normalize the similarity matrix
                    norm = torch.mm(
                        torch.norm(fake_flat, dim=1).unsqueeze(1),
                        torch.norm(fake_flat, dim=1).unsqueeze(0)
                    )
                    similarity_matrix = similarity_matrix / (norm + 1e-8)
                    
                    # We want off-diagonal elements to be small (dissimilar images)
                    # Create a mask for the off-diagonal elements
                    mask = 1.0 - torch.eye(batch_size, device=self.device)
                    
                    # Calculate penalty based on similarity
                    neighbor_penalty = torch.sum(mask * torch.abs(similarity_matrix)) / (batch_size * (batch_size - 1))
                    
                    # Apply the penalty to encourage diversity
                    similarity_stats = {
                        "mean": similarity_matrix.mean().item(),
                        "max": similarity_matrix.max().item(),
                        "min": similarity_matrix.min().item()
                    }
                    
                # Apply both losses to generator
                loss_G_total = loss_G + neighbor_penalty_lambda * neighbor_penalty
                
                # Actually apply the Generator loss gradient and update weights
                loss_G_total.backward()
                self.opt_G.step()
                
                # Record losses for logging
                G_losses.append(loss_G.item())
                D_losses.append(loss_D.item())
                
                batch_end = time.time()
                batch_times.append(batch_end - data_loaded)
                batch_start = time.time()
            
            # End of batch for-loop
            
            # Compute average losses, ignoring NaNs
            # Disable anomaly detection to restore performance
            torch.autograd.set_detect_anomaly(False)
            avg_D_loss = (sum([x for x in D_losses if not (isinstance(x, float) and x != x)]) / max(1, len([x for x in D_losses if not (isinstance(x, float) and x != x)]))) if any([not (isinstance(x, float) and x != x) for x in D_losses]) else float('nan')
            avg_G_loss = (sum([x for x in G_losses if not (isinstance(x, float) and x != x)]) / max(1, len([x for x in G_losses if not (isinstance(x, float) and x != x)]))) if any([not (isinstance(x, float) and x != x) for x in G_losses]) else float('nan')
            
            # Log average distortion for this epoch
            if distortion_values:
                avg_distortion = sum(distortion_values) / len(distortion_values)
                print(f"[DISTORTION][E{epoch+1}] Avg distortion: {avg_distortion:.4f} (manual_scale={manual_distortion_scale})")
            
            epoch_end = time.time()
            
            # Only one of D or G is paused per batch; sum equals number of batches
            num_batches = D_paused_batches + G_paused_batches
            print(f"[E{epoch+1}] D: {avg_D_loss:.4f} | G: {avg_G_loss:.4f} | D-paused: {D_paused_batches}/{num_batches} (streak {D_paused_streak}) | G-paused: {G_paused_batches}/{num_batches} (streak {G_paused_streak}) | Time: {epoch_end-epoch_start:.1f}s")
            
            # --- Fix: Avoid ZeroDivisionError if no batches processed ---
            avg_data_time = sum(data_times)/len(data_times) if data_times else 'N/A'
            avg_batch_time = sum(batch_times)/len(batch_times) if batch_times else 'N/A'
            print(f"[TIMING] Epoch {epoch+1}: Total {epoch_end-epoch_start:.2f}s | Avg data load {avg_data_time}s | Avg train {avg_batch_time}s per batch")
            print(f"[LR] G: {self.opt_G.param_groups[0]['lr']:.6f} | D: {self.opt_D.param_groups[0]['lr']:.6f}")
            
            # Save sample and checkpoint at intervals
            if (epoch + 1) % self.sample_interval == 0 or epoch == 0:
                print(f"[Epoch {epoch+1}] Saving sample and checkpoint...")
                self.generate_sample(epoch+1)
                self.save_checkpoint(epoch)
                # Also update the manual sample image (now on the same schedule as checkpoints)
                self.G.eval()
                with torch.no_grad():
                    fake = self.G(self.fixed_noise[:1]).detach().cpu()
                    save_image(fake, os.path.join(SAMPLE_DIR, 'sample_epochmanual.png'), normalize=True)
                self.G.train()
                
                # --- Save a grid of real images as seen by the Discriminator ---
                try:
                    real_batch = next(iter(train_loader))
                    real_grid_path = os.path.join(SAMPLE_DIR, 'real_grid_epoch.png')
                    save_image(real_batch[:16], real_grid_path, normalize=True, nrow=4, padding=2, pad_value=1)
                    # Log the average distortion used this epoch in the real grid log
                    if distortion_values:
                        avg_dist = sum(distortion_values) / len(distortion_values)
                        print(f"[SAMPLE] Saved real image grid: {real_grid_path} (avg distortion: {avg_dist:.4f})")
                    else:
                        print(f"[SAMPLE] Saved real image grid: {real_grid_path}")
                except Exception as e:
                    print(f"[ERROR] Failed to save real image grid: {e}")
            
            # Create full model file every git_push_interval epochs only
            if (epoch + 1) % self.git_push_interval == 0:
                self.save_full_model(epoch)
            
            # Save grid image with epoch in name every 1000 epochs
            if (epoch + 1) % 1000 == 0:
                try:
                    self.G.eval()
                    with torch.no_grad():
                        fake_grid = self.G(self.fixed_noise).detach().cpu()
                        grid_path = os.path.join(SAMPLE_DIR, f'sample_grid_epoch{epoch+1}.png')
                        save_image(fake_grid, grid_path, normalize=True, nrow=4, padding=2, pad_value=1)
                        print(f"[SAMPLE] Saved grid image: {grid_path}")
                    self.G.train()
                except Exception as e:
                    print(f"[ERROR] Failed to save grid image for epoch {epoch+1}: {e}")
                
                # Save full generator model every 1000 epochs with t_epoch{epoch+1}.pth
                full_model_path = os.path.join(DATA_DIR, f'gan_full_model_t_epoch{epoch+1}.pth')
                torch.save({
                    'G': self.G.state_dict(),
                    'opt_G': self.opt_G.state_dict(),
                    'epoch': epoch,
                    'img_size': self.img_size,
                    'resolution_history': self.resolution_history
                }, full_model_path)
                print(f"[FULLMODEL] Saved full generator model: {full_model_path}")
            
            # After epoch: pixel-perfect check
            if self.reference_image_tensor is not None:
                self.G.eval()
                with torch.no_grad():
                    gen_img = self.G(self.fixed_seed_noise)
                real_img = self.reference_image_tensor.to(self.device)
                if self._images_equal(gen_img, real_img):
                    print(f"[MATCH] Generator output matches reference image at epoch {epoch+1}!")
                    self._save_pixelmatch_outputs(epoch, gen_img, real_img)
                    print("[MATCH] Stopping training early.")
                    return  # Stop training
            
            epoch += 1
            vram_retry = 0  # Reset VRAM retry counter after successful epoch
        
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print(f"[VRAM] CUDA out of memory detected. Attempting VRAM fallback (attempt {vram_retry+1}/10)...")
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                self.checkpointing_level += 1
                self.enable_gradient_checkpointing()
                print(f"[VRAM] Aggressiveness level: {self.checkpointing_level}")
                vram_retry += 1
                if vram_retry > 10:
                    print("[VRAM] Too many VRAM fallback attempts. Exiting training.")
                    raise
                continue  # Retry epoch with more aggressive checkpointing
            else:
                raise
    
    print("Training complete.")
